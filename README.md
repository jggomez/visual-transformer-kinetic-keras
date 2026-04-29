# EuroSAT Vision Transformer (ViT) with Keras & Kinetic

This project implements a **Vision Transformer (ViT)** from scratch using **Keras 3** and **TensorFlow**, optimized for remote execution on Google Cloud Platform via **Kinetic**. The model is trained on the **EuroSAT** dataset (satellite imagery for land use and land cover classification).

## Architecture Overview

The model follows the original ViT architecture, where images are treated as sequences of patches.

<img width="3581" height="7440" alt="achitecture-dl" src="https://github.com/user-attachments/assets/1780e0a9-4a4e-458e-9f96-d60d1fb3cfe6" />

### Key Components:
- **Patch Embedding**: Uses a `Conv2D` layer as a hybrid stem to project image patches into a high-dimensional latent space.
- **Learnable [class] Token**: Prepended to the sequence of patch embeddings to capture global image representation.
- **Sinusoidal Position Encodings**: Injected into the sequence to provide spatial context.
- **Transformer Blocks**: Multi-head self-attention mechanisms with residual connections and layer normalization.

<img width="2152" height="1984" alt="Gemini_Generated_Image_hynpuchynpuchynp" src="https://github.com/user-attachments/assets/4c41654a-9f51-4229-8904-020a20686c9d" />

Image based on this [article](https://medium.com/machine-intelligence-and-deep-learning-lab/vit-vision-transformer-cc56c8071a20)

## Getting Started

This project uses `uv` for package management and `Kinetic` for GKE-accelerated training.

### 1. Prerequisites
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured.
- [Kinetic CLI](https://kinetic.readthedocs.io/) installed.
- Access to a GCP project with **NVIDIA L4 GPU** quota.

### 2. Infrastructure Management
Create or refresh the cluster and accelerator pools:
```bash
uv run kinetic up --project=YOUR_PROJECT_ID --zone=us-central1-c --accelerator=l4 --yes
```

To tear down the infrastructure and stop costs:
```bash
uv run kinetic down --yes
```

### 3. Running Training
To launch the training job on a remote NVIDIA L4 GPU:
```bash
uv run src/keras_kinetic_multivariate_analysis_clustering_eurosat_.py
```

## Kinetic Operations

### Monitor Logs
Once the job is submitted, you can stream the logs in real-time:
```bash
# Find your Job ID
kinetic jobs list

# Stream logs (replace <JOB_ID> with yours, e.g., job-3ebac5e5)
kinetic jobs logs <JOB_ID> --follow
```

### Manage Jobs
- **Check Status**: `kinetic jobs status <JOB_ID>`
- **Cancel Job**: `kinetic jobs cancel <ID>`
- **Cleanup Resources**: `kinetic jobs cleanup <ID> --k8s --gcs`

## Dataset: EuroSAT
- **Samples**: 27,000 labeled images.
- **Classes**: 10 (Forest, Industrial, Residential, Herbaceous Vegetation, Pasture, Highway, River, Lake, Annual Crop, Permanent Crop).
- **Format**: 64x64 RGB images.

## Configuration
- **Accelerator**: `l4`
- **Backend**: `tensorflow` (Keras 3)
- **Container Mode**: `bundled` (automatic image building)

---
Developed by Devhack - Juan G Gomez
