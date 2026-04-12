# 🛰️ Project Rover

## Purpose
Rover is a high-performance semantic segmentation system developed for the Duality AI Offroad Autonomy Challenge. The project utilizes synthetic data from the Falcon digital twin platform to train a model capable of navigating complex desert environments. The primary objective is to enhance UGV (Unmanned Ground Vehicle) path planning by providing fine-grained scene understanding, specifically targeting the detection of small-scale hazards that are often overlooked in traditional off-road datasets.

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
