# EuroSAT Vision Transformer (ViT) with Keras & Kinetic

<img width="2816" height="1536" alt="Gemini_Generated_Image_hbwxx6hbwxx6hbwx" src="https://github.com/user-attachments/assets/0e231f7a-54dd-4b86-a121-00249a83ffa6" />

This project implements a **Vision Transformer (ViT)** from scratch using **Keras 3** and **TensorFlow**, optimized for remote execution on Google Cloud Platform via **Kinetic**. The model is trained on the **EuroSAT** dataset (satellite imagery for land use and land cover classification).

### Notebook
[eurosat-vit-keras](https://colab.research.google.com/drive/1LC0zhHQxWH3_c0U22EfNBugQ2Fhn8iZq?usp=sharing)

## Blog Post
[From Local Prototyping to GCP Cloud GPUs: Building a Land Cover Classifier with Kinetic Keras](https://jggomezt.medium.com/from-local-prototyping-to-gpus-in-the-gcp-cloud-creating-a-satellite-image-classification-system-e280fc91fe67)

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

### 4. Results

Ultimately, the model achieved a highly robust validation accuracy of 92.83% (val_accuracy: 0.9283) with a validation loss of 0.2239 (loss: 0.2184 on the training set).

<img width="552" height="263" alt="Screenshot 2026-04-29 at 1 06 36 p m" src="https://github.com/user-attachments/assets/c4369fcf-c8ad-4153-80db-ae7395a90f74" />

Evaluating the detailed classification report on the 5,400 test images, the ViT achieved an overall macro and weighted average F1-score of 0.91 and 0.92, respectively. The model was remarkably precise at identifying distinct natural landscapes. For instance, the SeaLake class achieved a near-perfect F1-score of 0.98 with a precision of 1.00, while the Forest class followed closely with an impressive F1-score of 0.97. The Residential and HerbaceousVegetation categories also performed excellently, both scoring an F1-score of 0.93.

<img width="1072" height="999" alt="confusion" src="https://github.com/user-attachments/assets/697c6b84-ae98-4ed9-8d86-0b63148fd781" />

## References

- P. Helber, B. Bischke, A. Dengel, and D. Borth, "Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification," IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
- A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in International Conference on Learning Representations (ICLR), 2020.
- Keras Team, "keras-team/kinetic: Run ML workloads seamlessly on cloud TPUs and GPUs with a single Python decorator," GitHub, 2026. [Online]. Available: https://github.com/keras-team/kinetic.
- Keras Team, "Architecture Overview — kinetic documentation," 2026. [Online]. Available: https://kinetic.readthedocs.io.
- Keras Team, "Execution Modes — kinetic documentation," 2026. [Online]. Available: https://kinetic.readthedocs.io.
- Keras Team, "Training Keras Models — kinetic documentation," 2026. [Online]. Available: https://kinetic.readthedocs.io.
- Keras Team, "Accelerator Support — kinetic documentation," 2026. [Online]. Available: https://kinetic.readthedocs.io.
- G. Dahiya et al., "EuroSat Dataset," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset.
- Machine Intelligence and Deep Learning Lab, "ViT (Vision Transformer)," Medium. [Online]. Available: https://medium.com/machine-intelligence-and-deep-learning-lab/vit-vision-transformer-cc56c8071a20.
- J. G. Gómez Torres, "multivariate-analysis-clustering-eurosat," GitHub, 2026. [Online]. Available: https://github.com/jggomez/multivariate-analysis-clustering-eurosat

---
Developed by the EuroSAT Visual Transformer Team.
