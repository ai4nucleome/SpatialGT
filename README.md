# SpatialGT

SpatialGT is a graph transformer framework for spatial transcriptomics
reconstruction and virtual perturbation analysis. The public repository is
organized around the **SpatialGT Agent UI**, which provides an interactive
workflow for loading user-provided spatial transcriptomics data, preprocessing
the section, configuring perturbations, running SpatialGT inference, and
reviewing results.

Repository: [https://github.com/ai4nucleome/SpatialGT](https://github.com/ai4nucleome/SpatialGT)

## Agent UI

The recommended entry point is the Streamlit Agent UI in
`spatialgt_agent_ui/`. The UI supports both natural-language control and manual
configuration panels.

Typical workflow in the Agent UI:

1. Load a user-provided `.h5ad` spatial transcriptomics file.
2. Run preprocessing and build the LMDB cache.
3. Optionally load a label CSV and select perturbation target spots.
4. Load a SpatialGT checkpoint from Hugging Face or a local path.
5. Upload a DEG file or define manual gene edits.
6. Run dual-line virtual perturbation inference.
7. Review convergence curves, selected step, and saved output paths.

See [`spatialgt_agent_ui/USER_GUIDE.md`](spatialgt_agent_ui/USER_GUIDE.md) for
the Agent UI launch and usage guide.

## Runtime Environment

A packaged SpatialGT runtime environment is provided on Hugging Face so users
can run the Agent UI without rebuilding the Python stack manually:

- [SpatialGT Environment](https://huggingface.co/Bgoood/SpatialGT-Environment)

After downloading `spatialgt_env.tar.gz`, place it under `environment/` in the
repository root and extract it:

```bash
mkdir -p environment
tar -xzf environment/spatialgt_env.tar.gz -C environment
```

Then launch the Agent UI:

```bash
bash spatialgt_agent_ui/run.sh
```

The launcher automatically detects `environment/spatialgt`. You can also pass a
custom compatible environment:

```bash
bash spatialgt_agent_ui/run.sh --spatialgt /path/to/environment/spatialgt
```

## Large Files And Models

Large runtime assets are hosted outside GitHub. Model checkpoints, gene
embeddings, example data, outputs, and packaged environments are intentionally
excluded from this repository.

Available Hugging Face resources include:

- [SpatialGT-Pretrained](https://huggingface.co/Bgoood/SpatialGT-Pretrained)
- [SpatialGT-GeneEmbedding](https://huggingface.co/Bgoood/SpatialGT-GeneEmbedding)
- [SpatialGT Environment](https://huggingface.co/Bgoood/SpatialGT-Environment)

## Repository Contents

- `spatialgt_agent_ui/`: Agent-driven Streamlit user interface.
- `pretrain/`: SpatialGT model and data bank utilities.
- `finetune/`: Finetuning utilities used by the Agent UI.
- `reconstruction/`: Reconstruction and denoising scripts.
- `perturbation/`: Public perturbation case-study scripts.
- `baseline/`: Baseline method dependencies.
- `requirements.txt`: Python dependency list for source installations.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

## Contact

For questions and issues, please open a GitHub issue or contact
[yxu662@connect.hkust-gz.edu.cn](mailto:yxu662@connect.hkust-gz.edu.cn).
