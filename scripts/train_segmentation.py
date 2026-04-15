import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}
n_classes = len(value_map)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        new_arr[arr == raw] = new
    return new_arr

# ==============================
# DATASET
# ==============================

class MaskDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.ids = os.listdir(self.image_dir)
        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file = self.ids[idx]

        img = Image.open(os.path.join(self.image_dir, file)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, file))

        # Convert labels FIRST
        mask = convert_mask(mask)

        # Resize image
        img = self.img_transform(img)

        # Resize mask (CRITICAL: use NEAREST)
        mask = Image.fromarray(mask)
        mask = mask.resize(self.img_size[::-1], resample=Image.NEAREST)

        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask

# ==============================
# MODEL
# ==============================

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

# ==============================
# MAIN
# ==============================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    batch_size = 4
    lr = 1e-5
    epochs = 10

    # MUST be divisible by 14
    w, h = 476, 266
    img_size = (h, w)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')

    train_ds = MaskDataset(train_dir, img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    backbone.eval()

    sample = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        out = backbone.forward_features(sample)["x_norm_patchtokens"]
    emb_dim = out.shape[2]

    model = SegHead(emb_dim, n_classes, h//14, w//14).to(device)

    print("Training from scratch...")

    weights = torch.tensor([0.5,1,1,1.2,1.2,1.5,2.5,3.0,1,1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\nStarting training...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.no_grad():
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]

            logits = model(feats)
            logits = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "finetuned_model.pth")
    print("\n✅ Saved finetuned_model.pth")

if __name__ == "__main__":
    main()