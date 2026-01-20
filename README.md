# SpatialGT: Spatial Graph Transformer for Spatial Transcriptomics

<p align="center">
  <img src="assets/SpatialGT_Arch_v2.drawio.png" width="800">
</p>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**SpatialGT** is a graph transformer model for spatial transcriptomics data analysis. It leverages spatial context through neighbor-aware attention mechanisms to enable:

- 🗺️ **Spatial Context Learning**: Pre-train on large-scale spatial transcriptomics data
- 🧬 **Gene Expression Reconstruction**: Predict masked gene expression from spatial context
- 🔬 **Perturbation Simulation**: Simulate transcriptomic responses to virtual perturbations

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Pretraining](#pretraining)
  - [Finetuning](#finetuning)
  - [Reconstruction](#reconstruction)
  - [Perturbation Simulation](#perturbation-simulation)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)
- [License](#license)

## Installation

### Option 1: Conda (Recommended)

```bash
git clone https://github.com/ai4nucleome/SpatialGT.git
cd SpatialGT

# Create and activate environment
conda env create -f env/environment.yml
conda activate spatialgt
```

### Option 2: Pip

```bash
git clone https://github.com/ai4nucleome/SpatialGT.git
cd SpatialGT

# Install PyTorch with CUDA 11.8
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt
```

See [env/INSTALL.md](env/INSTALL.md) for detailed installation instructions.

## Quick Start

```python
import torch
from pretrain.model_spatialpt import SpatialNeighborTransformer
from pretrain.Config import Config

# Load configuration
config = Config()

# Initialize model
model = SpatialNeighborTransformer(config)

# Load pretrained weights
checkpoint = torch.load("path/to/checkpoint.pt")
model.load_state_dict(checkpoint)

# Your spatial transcriptomics data
# ...
```

## Repository Structure

```
SpatialGT/
├── pretrain/                 # Pretraining module
│   ├── Config.py            # Configuration
│   ├── model_spatialpt.py   # Model architecture
│   ├── spatial_databank.py  # Data loading utilities
│   ├── run_pretrain.py      # Training script
│   └── run.sh               # Launch script
│
├── finetune/                 # Finetuning module
│   ├── Config.py            # Finetuning configuration
│   ├── finetune.py          # Finetuning script
│   └── finetune.sh          # Launch script
│
├── reconstruction/           # Expression reconstruction
│   ├── spatialgt_reconstruction.py
│   ├── knn_reconstruction.py      # KNN baseline
│   ├── sedr_reconstruction.py     # SEDR baseline
│   └── run_reconstruction.sh
│
├── perturbation/             # Perturbation simulation
│   ├── mouse_stroke/        # Mouse stroke case study
│   └── human_colitis/       # Human colitis case study
│
├── gene_embedding/           # Pretrained gene embeddings (download from HuggingFace)
│   ├── vocab.json
│   ├── id_to_gene.json
│   └── pretrained_gene_embeddings.pt
│
├── baseline/                 # Baseline methods
│   └── SEDR/                # SEDR implementation
│
├── env/                      # Environment setup
│   ├── environment.yml
│   └── INSTALL.md
│
└── requirements.txt
```

## Data Preparation

### 1. Prepare H5AD Files

Your spatial transcriptomics data should be in AnnData format (`.h5ad`) with:
- `adata.X`: Gene expression matrix (cells × genes)
- `adata.obsm['spatial']`: Spatial coordinates
- `adata.var_names`: Gene symbols

### 2. Preprocess Data

```bash
# Preprocess your data for training
python pretrain/preprocess.py \
    --dataset_list /path/to/datalist.txt \
    --cache_dir /path/to/cache \
    --n_neighbors 8
```

The `datalist.txt` should contain paths to your H5AD files, one per line.

## Usage

### Pretraining

```bash
cd pretrain

# Single GPU
python run_pretrain.py \
    --dataset_list /path/to/datalist.txt \
    --output_dir /path/to/output

# Multi-GPU (distributed)
bash run.sh
```

### Finetuning

```bash
cd finetune

# Finetune on your dataset
bash finetune.sh \
    --base_ckpt /path/to/pretrained/checkpoint \
    --cache_dir /path/to/your/data \
    --output_dir /path/to/output
```

Key parameters:
- `--unfreeze_last_n`: Number of transformer layers to unfreeze (default: 8, all layers)
- `--num_epochs`: Training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-4)

### Reconstruction

```bash
cd reconstruction

# SpatialGT reconstruction (10 steps)
bash run_reconstruction.sh --method spatialgt --n_spots 100

# SEDR baseline (1 step)
bash run_reconstruction.sh --method sedr --n_spots 100

# KNN baseline (10 steps)
bash run_reconstruction.sh --method knn --n_spots 100
```

### Perturbation Simulation

#### Mouse Stroke Case

```bash
cd perturbation/mouse_stroke

# Run perturbation with ICA region
bash run_perturbation.sh output_name \
    --perturb_mode random \
    --n_spots 80 \
    --steps 10
```

#### Human Colitis Case

```bash
cd perturbation/human_colitis

# Run perturbation on activated MNPs
python colitis_spatialgt_perturb_eval.py \
    --sample HS5_UC_R_0 \
    --perturb_target MNP_activated \
    --steps 10
```

## Pretrained Models

We provide pretrained and finetuned model checkpoints on Hugging Face:

| Model | Description | Download |
|-------|-------------|----------|
| SpatialGT-Pretrained | Pretrained on spatial transcriptomics atlas | [🤗 Hugging Face](https://huggingface.co/Bgoood/SpatialGT-Pretrained) |
| SpatialGT-MouseStroke-Sham | Finetuned on mouse stroke Sham (control) | [🤗 Hugging Face](https://huggingface.co/Bgoood/SpatialGT-MouseStroke-Sham) |
| SpatialGT-MouseStroke-PT | Finetuned on mouse stroke PT (stroke) | [🤗 Hugging Face](https://huggingface.co/Bgoood/SpatialGT-MouseStroke-PT) |
| SpatialGT-GeneEmbedding | Pretrained gene embeddings | [🤗 Hugging Face](https://huggingface.co/Bgoood/SpatialGT-GeneEmbedding) |

### Download Models

```bash
# Using huggingface-cli
huggingface-cli download Bgoood/SpatialGT-Pretrained --local-dir model/pretrain_ckpt
huggingface-cli download Bgoood/SpatialGT-MouseStroke-Sham --local-dir model/sham_1_ft
huggingface-cli download Bgoood/SpatialGT-MouseStroke-PT --local-dir model/pt_ft

# Download pretrained gene embeddings
huggingface-cli download Bgoood/SpatialGT-GeneEmbedding --local-dir gene_embedding
```

Or using Python:

```python
from huggingface_hub import snapshot_download

# Download pretrained model
snapshot_download(repo_id="Bgoood/SpatialGT-Pretrained", local_dir="model/pretrain_ckpt")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and issues, please open a GitHub issue or contact [yxu662@connect.hkust-gz.edu.cn](mailto:yxu662@connect.hkust-gz.edu.cn).

## Acknowledgments

- [Scanpy](https://scanpy.readthedocs.io/) for single-cell analysis tools
- [Hugging Face Transformers](https://huggingface.co/transformers/) for transformer implementations
- [SEDR](https://github.com/JinmiaoChenLab/SEDR) for the baseline method
