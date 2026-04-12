# 🛰️ Project Rover

## The Problem: Off-Road UGV Navigation
Navigating complex desert environments poses a unique challenge for Unmanned Ground Vehicles (UGVs). Traditional off-road datasets often overlook small-scale, deeply embedded hazards like rocks and logs that blend into the surrounding landscape. If a rover misclassifies a 10-inch rock as navigable sand, it risks severe mechanical damage (e.g., broken axles) and mission failure in remote, difficult-to-access areas.

## The Solution: Project Rover
Rover is a high-performance semantic segmentation system developed for the Duality AI Offroad Autonomy Challenge. By utilizing high-fidelity synthetic data sourced from the Falcon digital twin platform, Rover trains a highly specialized model designed to detect and outline off-road hazards with pixel-perfect accuracy. The primary objective is to enhance UGV path planning by providing fine-grained scene understanding, ensuring the vehicle can distinguish between safe terrain and critical obstacles in real-time.

## Technical Glossary
To ensure technical rigor, the project utilizes the following concepts:
*   **Semantic Scene Segmentation:** The process of labeling every single pixel in an image with a specific class (e.g., "Rock" vs. "Sand"), providing a complete environmental mask rather than simple bounding boxes.
*   **Intersection over Union (IoU):** The standard "Overlap Accuracy" metric used to evaluate pixel classification. It measures how perfectly the predicted mask aligns with the actual object in the image.
*   **Weighted Loss (15x Penalty):** A custom training strategy designed to address class imbalance. In our model, failing to detect a critical obstacle (like a rock) incurs a penalty 15 times higher than misclassifying a non-critical feature (like the sky).
*   **Digital Twin Simulation:** The use of high-quality, physics-accurate synthetic environments to train AI for remote or dangerous areas without risking physical hardware.

## Tactical Class Mapping & Scene Visualization
The neural engine is trained to segment specific environmental classes. During inference, the system maps each identified class number to a specific pixel color to generate the diagnostic overlay mask. This provides human operators with an instant visual understanding of the UGV's surroundings:

| Class ID | Class Name | Strategic Importance | Diagnostic Mask Color |
| :--- | :--- | :--- | :--- |
| **800** | Rocks | Critical Hazard | 🔴 Red |
| **700** | Logs | Critical Hazard | 🔴 Red |
| **100-600** | Vegetation | Complex Obstacles | 🟡 Yellow / Amber |
| **7100** | Landscape | Navigable Ground | 🟢 Green / Emerald |
| **10000** | Sky | Environmental Context | 🔵 Blue / Cyan |

## Architecture & Tech Stack

**Core Architecture**
Rover employs a decoupled, modular architecture designed for edge-compute efficiency and scalability. 

**Tech Stack**
*   **Neural Engine (Planned):** DINOv2 (Vision Transformer) backbone for advanced feature extraction.
*   **Backend:** Pure Python FastAPI server managing the RESTful neural bridge. It securely serves the diagnostic frontend and acts as an intelligent socket for image processing endpoints without heavy Node.js or framework overhead.
*   **Frontend:** The *Rover Dark Diagnostic Interface*. A 100% Python-served, single-file Vanilla JS and Tailwind CSS dashboard. It visualizes the segmentation process natively in the browser while maintaining rigorous AMOLED aesthetics and structural layouts.

## Pending Work / Development Roadmap

*   **Phase 1 (AI Integration):** Replace the placeholder logic in the FastAPI `/scan` route with actual PyTorch model inference payloads. 
*   **Phase 2 (Training Cycle):** Execute training on the local ROG node using the provided Duality AI training and validation sets. Apply Weighted Loss (15x Penalty) tuning to handle class imbalances for critical hazards.
*   **Phase 3 (Validation & Testing):** Test the model on unseen out-of-distribution desert images to ensure generalization strength.
*   **Phase 4 (Reporting):** Finalize the performance report highlighting failure case analysis and final intersection validation metrics. 
