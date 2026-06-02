# SpatialGT: Spatial Graph Transformer for Spatial Transcriptomics

<p align="center">
  <img src="assets/SpatialGT_Arch_v2.drawio.png" width="800">
</p>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-SpatialGT--Pretrained-orange)](https://huggingface.co/Bgoood/SpatialGT-Pretrained)
[![HuggingFace Embedding](https://img.shields.io/badge/%F0%9F%A4%97%20Embedding-SpatialGT--GeneEmbedding-orange)](https://huggingface.co/Bgoood/SpatialGT-GeneEmbedding)
[![HuggingFace Environment](https://img.shields.io/badge/%F0%9F%A4%97%20Environment-SpatialGT--Environment-orange)](https://huggingface.co/Bgoood/SpatialGT-Environment)

## Overview

**SpatialGT** is a graph transformer model for spatial transcriptomics data
analysis. It leverages spatial context through neighbor-aware attention to enable:

- **Spatial Context Learning**: pre-train on large-scale spatial transcriptomics data.
- **Gene Expression Reconstruction**: predict masked gene expression from spatial context.
- **Virtual Perturbation Simulation**: simulate transcriptomic responses to in-silico perturbations.

The public repository is organized around the **SpatialGT Agent UI**, an
interactive application that lets you load your own spatial transcriptomics
data, preprocess a section, configure perturbations, run SpatialGT inference,
and review results — either through natural-language agent control or manual
configuration panels.

## Table of Contents

- [Agent UI](#agent-ui)
- [Environment Setup](#environment-setup)
  - [Option 1: Packaged Environment (Recommended)](#option-1-packaged-environment-recommended)
  - [Option 2: Build From Scratch](#option-2-build-from-scratch)
- [Models And Embeddings](#models-and-embeddings)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Contact](#contact)

## Agent UI

The recommended entry point is the Streamlit Agent UI in `spatialgt_agent_ui/`.
It exposes both an LLM **Agent** page for natural-language control and manual
pages for step-by-step configuration. See
[`spatialgt_agent_ui/USER_GUIDE.md`](spatialgt_agent_ui/USER_GUIDE.md) for the
full launch and usage guide.

### Launch The UI

From the repository root, after preparing an environment (see below):

```bash
bash spatialgt_agent_ui/run.sh
```

The launcher automatically detects `environment/spatialgt`. You can also pass a
custom compatible environment and port:

```bash
bash spatialgt_agent_ui/run.sh --spatialgt /path/to/environment/spatialgt --port 8501
```

Then open `http://SERVER_IP:8501`. For remote servers, forward the port over SSH:

```bash
ssh -L 8501:localhost:8501 user@server
```

A Docker workflow is also available from `spatialgt_agent_ui/`:

```bash
docker compose up --build
```

### Configure The Agent

To use natural-language control, provide an LLM API key:

```bash
cd spatialgt_agent_ui
cp .env.example .env
# edit .env and set SILICONFLOW_API_KEY (do not commit .env)
```

With the agent configured, you can drive the entire workflow from the `Agent`
page in plain language, for example: "load this h5ad file", "preprocess and
build the cache", "select target spots by label", "load the pretrained model
from Hugging Face", "apply this DEG file", and "run perturbation inference".

### Typical Workflow

1. Load a user-provided `.h5ad` spatial transcriptomics file.
2. Run preprocessing and build the LMDB cache.
3. Optionally load a label CSV and select perturbation target spots.
4. Load a SpatialGT checkpoint from Hugging Face or a local path.
5. Upload a DEG file or define manual gene edits.
6. Run dual-line virtual perturbation inference.
7. Review convergence curves, the selected step, and saved output paths.

No test dataset is bundled with the UI; you provide your own `.h5ad`, optional
label CSV, and optional DEG file.

## Environment Setup

### Option 1: Packaged Environment (Recommended)

A ready-to-run SpatialGT runtime environment is hosted on Hugging Face so you
can launch the Agent UI without rebuilding the Python stack manually:

[![HuggingFace Environment](https://img.shields.io/badge/%F0%9F%A4%97%20Download-SpatialGT--Environment-orange)](https://huggingface.co/Bgoood/SpatialGT-Environment)

Download `spatialgt_env.tar.gz`, place it under `environment/` in the
repository root, and extract it:

```bash
mkdir -p environment
huggingface-cli download Bgoood/SpatialGT-Environment spatialgt_env.tar.gz \
    --repo-type model --local-dir environment
tar -xzf environment/spatialgt_env.tar.gz -C environment
```

After extraction, `environment/spatialgt/bin/python` is detected automatically
by the launcher:

```bash
bash spatialgt_agent_ui/run.sh
```

### Option 2: Build From Scratch

If you prefer to build your own environment:

```bash
git clone https://github.com/ai4nucleome/SpatialGT.git
cd SpatialGT

# Create and activate a Python 3.8+ environment (conda example)
conda create -n spatialgt python=3.8 -y
conda activate spatialgt

# Install PyTorch with CUDA 11.8
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install core dependencies
pip install -r requirements.txt

# Install Agent UI dependencies
pip install -r spatialgt_agent_ui/requirements.txt
```

Then launch the UI against this environment:

```bash
bash spatialgt_agent_ui/run.sh --spatialgt /path/to/your/conda/envs/spatialgt
```

## Models And Embeddings

Pretrained and finetuned checkpoints, along with gene embeddings, are hosted on
Hugging Face:

| Resource | Description | Link |
|----------|-------------|------|
| SpatialGT-Pretrained | Pretrained on the spatial transcriptomics atlas | [🤗 Hugging Face](https://huggingface.co/Bgoood/SpatialGT-Pretrained) |
| SpatialGT-MouseStroke-Sham | Finetuned on mouse stroke Sham (control) | [🤗 Hugging Face](https://huggingface.co/Bgoood/SpatialGT-MouseStroke-Sham) |
| SpatialGT-MouseStroke-PT | Finetuned on mouse stroke PT (stroke) | [🤗 Hugging Face](https://huggingface.co/Bgoood/SpatialGT-MouseStroke-PT) |
| SpatialGT-GeneEmbedding | Pretrained gene embeddings | [🤗 Hugging Face](https://huggingface.co/Bgoood/SpatialGT-GeneEmbedding) |
| SpatialGT-Environment | Packaged runtime environment | [🤗 Hugging Face](https://huggingface.co/Bgoood/SpatialGT-Environment) |

Download with the Hugging Face CLI:

```bash
# Pretrained model and gene embeddings
huggingface-cli download Bgoood/SpatialGT-Pretrained --local-dir model/pretrain_ckpt
huggingface-cli download Bgoood/SpatialGT-GeneEmbedding --local-dir gene_embedding
```

Or from Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Bgoood/SpatialGT-Pretrained", local_dir="model/pretrain_ckpt")
snapshot_download(repo_id="Bgoood/SpatialGT-GeneEmbedding", local_dir="gene_embedding")
```

The Agent UI can also download a checkpoint from Hugging Face directly when you
load a model.

## Repository Structure

- `spatialgt_agent_ui/`: Agent-driven Streamlit user interface.
- `pretrain/`: SpatialGT model architecture and data bank utilities.
- `finetune/`: Finetuning utilities used by the Agent UI.
- `reconstruction/`: Expression reconstruction and denoising scripts.
- `perturbation/`: Public perturbation case-study scripts.
- `baseline/`: Baseline method dependencies.
- `assets/`: Figures and static assets.
- `requirements.txt`: Python dependency list for source installations.

Large assets (model checkpoints, gene embeddings, example data, outputs, and
packaged environments) are hosted on Hugging Face and intentionally excluded
from this repository.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

## Contact

For questions and issues, please open a GitHub issue or contact
[yxu662@connect.hkust-gz.edu.cn](mailto:yxu662@connect.hkust-gz.edu.cn).
