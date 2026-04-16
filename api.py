from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import time
import uvicorn
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import io
import base64
import numpy as np
import torchvision.transforms as transforms
import cv2

# ============================================================================
# MODEL DEFINITION
# This SegHead architecture MUST match the architecture used during training.
# Using any other architecture (e.g. ConvNeXt head) will cause weight loading
# failures and incorrect predictions.
# ============================================================================
class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels, H, W):
        super().__init__()
        self.H, self.W = H, W

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.net(x)

# ============================================================================
# APP INITIALIZATION
# ============================================================================
app = FastAPI(title="AuraNav Edge API")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Input resolution: must be divisible by 14 (DINOv2 patch size) and must
# match the resolution used during training exactly to avoid patch misalignment.
w, h = 476, 266

print("Loading DINOv2 backbone...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval().to(device)

print("Loading fine-tuned segmentation head...")
# finetuned_model.pth was produced by train_segmentation_optimized.py and is
# the correct weights file. Do NOT use best_segmentation_head.pth here as that
# file uses a different output head shape.
head = SegHead(in_channels=384, out_channels=10, H=h//14, W=w//14)
head.load_state_dict(torch.load("finetuned_model.pth", map_location=device))
head.eval().to(device)

# Preprocessing pipeline - identical to the one used in training
transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Color palette: one RGB color per class index (0-9)
color_palette = np.array([
    [0,   0,   0  ],  # 0 - Background
    [34,  139, 34 ],  # 1 - Trees
    [50,  205, 50 ],  # 2 - Lush Bushes
    [255, 215, 0  ],  # 3 - Dry Grass
    [255, 140, 0  ],  # 4 - Dry Bushes
    [128, 128, 128],  # 5 - Ground Clutter
    [255, 0,   255],  # 6 - Logs (hazard)
    [255, 0,   0  ],  # 7 - Rocks (hazard)
    [0,   255, 0  ],  # 8 - Landscape
    [0,   191, 255],  # 9 - Sky
], dtype=np.uint8)

# ============================================================================
# ROUTES
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the diagnostic frontend interface."""
    with open("diagnostic_interface.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs DINOv2 + SegHead inference, and returns:
    - mask_base64: the blended segmentation overlay as a JPEG base64 string
    - telemetry: hazard percentage and inference latency
    """
    try:
        start_time = time.time()

        # Read and decode uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess image to match training pipeline
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            feats = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
            logits = head(feats)

            # Upsample back to training resolution
            logits = F.interpolate(
                logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )

            # The model tends to over-predict class 7 (Rocks) due to class
            # imbalance in the dataset. Subtracting a fixed bias from the rock
            # logit channel reduces this without retraining.
            logits[:, 7, :, :] -= 1.2

            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Morphological opening removes small isolated noise pixels,
        # producing cleaner class region boundaries.
        kernel = np.ones((5, 5), np.uint8)
        pred_mask = cv2.morphologyEx(
            pred_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
        )

        # Compute hazard telemetry
        total_pixels = pred_mask.size
        rock_pixels = np.sum(pred_mask == 7)
        log_pixels  = np.sum(pred_mask == 6)
        hazard_percent = ((rock_pixels + log_pixels) / total_pixels) * 100

        # Colorize prediction and blend with original image
        color_mask = color_palette[pred_mask]
        base_img  = image.copy().convert("RGB")
        mask_img  = Image.fromarray(color_mask).resize(base_img.size, Image.NEAREST)
        final_hud = Image.blend(base_img, mask_img, alpha=0.5)

        # Encode blended image as JPEG base64
        buffered = io.BytesIO()
        final_hud.save(buffered, format="JPEG", quality=85)
        mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "SUCCESS",
            "mask_base64": f"data:image/jpeg;base64,{mask_base64}",
            "telemetry": {
                "hazard_level": f"{hazard_percent:.1f}%",
                "latency": f"{latency_ms}ms"
            }
        }

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
