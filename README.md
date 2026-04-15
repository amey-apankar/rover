# Project Rover

## The Problem: Off-Road UGV Navigation

Navigating complex desert environments poses a unique challenge for Unmanned Ground Vehicles (UGVs). Traditional off-road datasets often overlook small-scale, deeply embedded hazards like rocks and logs that blend into the surrounding landscape. If a rover misclassifies a 10-inch rock as navigable sand, it risks severe mechanical damage (broken axles, wheel failures) and mission failure in remote, difficult-to-access areas.

## The Solution: Project Rover

Rover is a high-performance semantic segmentation system developed for the Duality AI Offroad Autonomy Challenge. It uses high-fidelity synthetic data sourced from the Falcon digital twin platform to train a specialized model designed to detect and outline off-road hazards with pixel-level accuracy. The primary objective is to enhance UGV path planning by providing fine-grained scene understanding, so the vehicle can distinguish between safe terrain and critical obstacles in real time.


---


## Quick Start

Clone this repository and install the dependencies:

```bash
git clone https://github.com/amey-apankar/rover.git
cd rover
pip install -r requirements.txt
```

Download the model weight file `finetuned_model.pth` and place it in the project root directory (see the Models section below for details).

Start the server:

```bash
python api.py
```

Open `http://127.0.0.1:8000` in your browser to access the diagnostic interface. Upload any off-road image to get a segmentation overlay with hazard telemetry.


---


## Key Concepts and Metrics

This section explains the technical terms used throughout the project documentation and codebase.

### Semantic Segmentation

Semantic segmentation assigns a class label to every single pixel in an image. Unlike object detection (which draws bounding boxes), segmentation produces a per-pixel mask where each pixel is classified as one of the defined classes (e.g., rock, tree, sky). This gives the rover a complete spatial understanding of its environment rather than just knowing that "there is a rock somewhere in the frame."

### IoU (Intersection over Union)

IoU, also called the Jaccard Index, is the standard metric for evaluating segmentation quality. For each class, it measures how well the predicted mask overlaps with the ground truth mask:

```
IoU = (Area of Overlap) / (Area of Union)
```

- An IoU of 1.0 means the prediction perfectly matches the ground truth.
- An IoU of 0.0 means zero overlap.
- In practice, IoU above 0.5 is considered acceptable for many applications.

Mean IoU (mIoU) averages the per-class IoU scores across all classes to give a single number summarizing model performance.

### mAP (Mean Average Precision)

mAP measures how well a model detects and classifies objects at various confidence thresholds. While mAP is more commonly used in object detection tasks, it is sometimes referenced in segmentation contexts when evaluating per-class detection reliability. A higher mAP means the model consistently identifies the correct class with high confidence.

### Dice Coefficient

The Dice coefficient (also called the F1 score for segmentation) is another overlap metric closely related to IoU:

```
Dice = (2 * Area of Overlap) / (Total pixels in prediction + Total pixels in ground truth)
```

Dice tends to produce higher numbers than IoU for the same prediction because it double-counts the overlap region. Both our best validation IoU (0.299) and Dice (0.443) are from epoch 14 of training.

### Weighted Cross-Entropy Loss

Standard cross-entropy loss treats all classes equally. In our dataset, hazard classes like Rocks and Logs occupy a small fraction of the total image area compared to dominant classes like Sky and Landscape. Without correction, the model simply learns to predict the majority class everywhere and ignores rare but critical hazards.

We address this by assigning higher penalties (weights) to under-represented hazard classes during training:

| Class         | Weight |
| :------------ | :----- |
| Background    | 0.5    |
| Trees         | 1.0    |
| Lush Bushes   | 1.0    |
| Dry Grass     | 1.2    |
| Dry Bushes    | 1.2    |
| Ground Clutter| 1.5    |
| Logs          | 2.5    |
| Rocks         | 3.0    |
| Landscape     | 1.0    |
| Sky           | 1.0    |

This forces the model to pay significantly more attention to detecting rocks and logs, even though they appear in fewer pixels overall. Without these weights, the model would achieve high overall accuracy by predicting sky and landscape everywhere while completely missing the hazards that matter most for navigation safety.


---


## Why We Fine-Tuned After Initial Training

The initial training run used the basic `train_segmentation.py` script with a simple weighted cross-entropy loss. While this produced a functional model, it had several problems:

1. **Rock overprediction.** The 3x weight on the rock class successfully made the model detect rocks, but it became too aggressive and started hallucinating rocks on sandy terrain and ground clutter. The model learned that predicting "rock" was safe because the loss penalty for missing a rock was high, so it over-indexed on that class.

2. **Noisy class boundaries.** The basic training script did not track validation metrics or use learning rate scheduling, so the model did not converge to clean decision boundaries between similar-looking classes (e.g., dry grass vs. ground clutter).

3. **No early stopping or checkpointing.** Without validation tracking, there was no way to know when the model started overfitting.

The optimized training script (`train_segmentation_optimized.py`) addressed all three issues:
- Added validation split with per-epoch IoU, Dice, and accuracy tracking.
- Added learning rate scheduling (cosine annealing) for smoother convergence.
- Saved the best model checkpoint based on validation IoU.
- The final `finetuned_model.pth` weights are from this optimized run.

At inference time, we also apply a post-training correction: subtracting 1.2 from the rock class logits before argmax. This acts as a confidence threshold that reduces false positive rock predictions without retraining the entire model. Combined with morphological opening (a small 5x5 kernel that removes isolated noise pixels), the result is a significantly cleaner segmentation output.


---


## Tactical Class Mapping and Scene Visualization

The model segments each image into 10 environment classes. During inference, each class is mapped to a distinct color for the diagnostic overlay:

| Class ID | Class Name      | Role                   | Overlay Color     |
| :------- | :-------------- | :--------------------- | :---------------- |
| 0        | Background      | Unclassified           | Black             |
| 1        | Trees           | Complex obstacle       | Forest Green      |
| 2        | Lush Bushes     | Complex obstacle       | Lime Green        |
| 3        | Dry Grass       | Moderate obstacle      | Gold              |
| 4        | Dry Bushes      | Moderate obstacle      | Dark Orange       |
| 5        | Ground Clutter  | Terrain noise          | Gray              |
| 6        | Logs            | Critical hazard        | Magenta           |
| 7        | Rocks           | Critical hazard        | Red               |
| 8        | Landscape       | Navigable ground       | Green             |
| 9        | Sky             | Environmental context  | Deep Sky Blue     |


---


## Architecture and Tech Stack

### Core Pipeline

```
Input Image --> DINOv2-ViTS14 Backbone --> Patch Token Features --> SegHead (Conv2d) --> Logit Map --> Argmax --> Class Mask
```

1. **DINOv2-ViTS14** (frozen backbone): A self-supervised Vision Transformer pre-trained by Meta on 142 million images. It produces dense patch-level feature embeddings (384-dimensional vectors for each 14x14 pixel patch). We do not fine-tune the backbone; we only train the lightweight head on top of it.

2. **SegHead** (trainable): A small convolutional network (Conv2d 384->128, GELU, Conv2d 128->128, GELU, Conv2d 128->10) that maps the DINOv2 patch features to per-class logits. This head is what gets saved as `finetuned_model.pth`.

3. **Post-processing**: Bilinear upsampling to input resolution, rock logit bias correction (-1.2), morphological opening (5x5 kernel), and color palette mapping.

### Tech Stack

- **Neural Engine:** DINOv2-ViTS14 backbone + custom SegHead (PyTorch)
- **Backend:** Python FastAPI server serving the REST inference endpoint and the diagnostic frontend
- **Frontend:** Single-file HTML/JS/CSS diagnostic dashboard served by the FastAPI backend, with real-time segmentation overlay display and hazard telemetry
- **Training:** Custom PyTorch training scripts with weighted loss, cosine LR scheduling, and validation checkpointing


---


## Training Results

Training was conducted for 20 epochs on the Duality AI Offroad Segmentation Training Dataset using the optimized script.

### Best Validation Metrics (Epoch 14)

| Metric          | Value  |
| :-------------- | :----- |
| Validation IoU  | 0.2990 |
| Validation Dice | 0.4432 |
| Validation Acc  | 0.6337 |

### Final Metrics (Epoch 20)

| Metric         | Train  | Validation |
| :------------- | :----- | :--------- |
| Loss           | 0.8946 | 0.9045     |
| IoU            | 0.3532 | 0.2955     |
| Dice           | 0.4640 | 0.4389     |
| Accuracy       | 0.6285 | 0.6281     |

Training curves and detailed per-epoch metrics are available in `scripts/train_stats/`.


---


## Project Structure

```
rover/
  api.py                        # FastAPI server with DINOv2 + SegHead inference
  diagnostic_interface.html     # Browser-based diagnostic UI
  finetuned_model.pth           # Trained SegHead weights (not tracked in git)
  requirements.txt              # Python dependencies
  prd                           # Product requirements document
  MISSION_LOG.md                # Development log
  README.md                     # This file
  scripts/
    train_segmentation.py           # Basic training script
    train_segmentation_optimized.py # Full training with validation and checkpointing
    test_segmentation.py            # Evaluation and metric computation
    visualize.py                    # Inference visualization utility
    ENV_SETUP/
      create_env.bat                # Create conda environment (Windows)
      install_packages.bat          # Install all packages (Windows)
      setup_env.bat                 # Activate environment (Windows)
    train_stats/
      evaluation_metrics.txt        # Per-epoch training and validation metrics
      training_curves.png           # Loss curves
      iou_curves.png                # IoU curves
      dice_curves.png               # Dice curves
      all_metrics_curves.png        # Combined metric curves
```


---


## Model Files

The `.pth` model weight files are not tracked in git due to their size. You need to obtain them separately:

| File                         | Size   | Description                                     |
| :--------------------------- | :----- | :---------------------------------------------- |
| `finetuned_model.pth`        | ~2.3MB | SegHead weights (required for inference)         |
| `best_segmentation_head.pth` | ~9.3MB | Full segmentation head checkpoint (training use) |

Place `finetuned_model.pth` in the project root directory before running `api.py`.


---


## Development Status

- [x] Phase 1 - AI Integration: DINOv2 + SegHead inference pipeline integrated into FastAPI
- [x] Phase 2 - Training: Model trained for 20 epochs with weighted loss on Duality AI dataset
- [x] Phase 3 - Validation and Testing: Evaluated on validation set with per-class IoU, Dice, and accuracy
- [x] Phase 4 - Reporting: Training metrics documented, convergence curves generated
