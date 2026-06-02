# SpatialGT Agent UI User Guide

This guide explains how to launch the SpatialGT Agent UI and run a user-provided spatial transcriptomics perturbation workflow. No test dataset is bundled with the UI; users must provide their own `.h5ad`, optional label CSV, and optional DEG file.

## Files

- `app.py`: Streamlit entry point.
- `run.sh`: bare-metal launcher.
- `Dockerfile`: Docker image definition.
- `docker-compose.yml`: Docker Compose launcher.
- `.env.example`: agent API configuration template.

## Configure The Agent API

Create a local `.env` if you want to use the LLM agent:

```bash
cp .env.example .env
```

Then edit `.env` and set `SILICONFLOW_API_KEY`. Do not commit `.env`.

## Bare-Metal Launch

From the `spatialgt_agent_ui` directory:

```bash
bash run.sh
```

The script first looks for a packaged environment at `../environment/spatialgt`. If it is not present, it falls back to `../../environment/spatialgt` for local development. You can also pass an explicit environment path:

```bash
bash run.sh --spatialgt /path/to/environment/spatialgt --port 8501
```

Open `http://SERVER_IP:8501` after launch. For SSH access, use port forwarding:

```bash
ssh -L 8501:localhost:8501 user@server
```

## Docker Launch

Build and start the UI from the `spatialgt_agent_ui` directory:

```bash
docker compose up --build
```

Run in the background:

```bash
docker compose up -d --build
```

Stop the service:

```bash
docker compose down
```

The default port is `8501`. User data is not mounted by default; add a bind mount in `docker-compose.yml` for your dataset directory and enter the container path in the UI.

## Recommended Workflow

1. Open the `Agent` page or use the manual pages from the sidebar.
2. Load a user-provided `.h5ad` file or server-side folder containing one `.h5ad`.
3. Run preprocessing and build the LMDB cache.
4. Optionally load a label CSV and select perturbation target spots.
5. Load a local checkpoint or download the pretrained model from Hugging Face.
6. Upload a DEG file or manually define gene edits.
7. Run dual-line perturbation inference.
8. Review convergence curves and saved output paths on the Results page.

## DEG File Format

Recommended columns: `gene`, `avg_logFC`, and optionally `p_val_adj`. Common aliases are also accepted, including `gene_name`, `symbol`, `logFC`, `log2FoldChange`, `fold_change`, `direction`, `padj`, and `fdr`. A one-column gene list is accepted and defaults to `avg_logFC = 1.0`.

## Outputs

Inference outputs are written under the selected cache workspace, usually `workspace/cache/perturbation_results`, including:

- `baseline_expressions.pkl`
- `step_001_expressions.pkl`, `step_002_expressions.pkl`, ...
- `perturbation_summary.json`

The Results page displays the selected step, MSE curves, summary path, and per-step expression paths.

## Troubleshooting

If imports fail, launch through `bash run.sh` with the correct SpatialGT environment. If CUDA is unavailable, verify GPU drivers and `CUDA_VISIBLE_DEVICES`. If inference is slow, make sure preprocessing completed in LMDB mode and reduce batch size if needed.
