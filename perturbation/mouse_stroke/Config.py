"""
SpatialGT Perturbation Configuration

Configuration for mouse stroke perturbation experiments.
Default: perturb ICA region, evaluate on ICA/PIA_P/PIA_D regions.
"""

import json
from pathlib import Path
import torch


class PerturbationConfig:
    """Configuration class for SpatialGT perturbation experiments."""
    
    def __init__(self):
        # ============ Path Configuration ============
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Gene embedding files
        self.vocab_file = str(self.project_root / "gene_embedding" / "vocab.json")
        self.pretrained_gene_embeddings_path = str(
            self.project_root / "gene_embedding" / "pretrained_gene_embeddings.pt"
        )
        
        # ============ Model Architecture ============
        self.model_name = "spatialgt"
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 8
        self.gene_encoder_layers = 6
        self.decoder_layers = 4
        self.dropout = 0.1
        
        # ============ Perturbation Parameters ============
        # Number of iteration steps (default: 10)
        self.steps = 10
        
        # Target ROI for perturbation (default: ICA)
        self.perturb_target_roi = "ICA"
        
        # Evaluation ROIs
        self.eval_rois = ["ICA", "PIA_P", "PIA_D"]
        
        # Perturbation mode: "patch" (BFS-connected) or "random"
        self.perturb_mode = "patch"
        self.patch_size = 20
        self.n_spots = 20
        
        # Weighting scheme: "gaussian" or "uniform"
        self.weighting = "gaussian"
        self.sigma = None  # Auto-computed if None
        
        # DEG-based perturbation
        self.p_adj_thresh = 0.1
        self.min_abs_logfc = 0.0
        self.logfc_strength = 1.0
        self.logfc_clip = 5.0
        
        # Freeze perturbed spots during iteration
        self.freeze_perturbed = True
        
        # ============ Spatial Configuration ============
        self.max_neighbors = 8
        self.distance_threshold = 1800
        self.adjacency_type = "value"
        
        # ============ Data Configuration ============
        self.cache_mode = "h5"
        self.lmdb_path = None
        self.lmdb_manifest_path = None
        self.lmdb_map_size_gb = 1024
        self.lmdb_max_readers = 1024
        
        # ============ Inference Configuration ============
        self.batch_size = 64
        self.num_workers = 4
        self.prefetch_factor = 2
        self.persistent_workers = False
        
        # ============ Device & Precision ============
        self.device = "cuda"
        self.dtype = torch.float16
        
        # ============ Reproducibility ============
        self.random_seed = 42
        
        # ============ Additional Parameters (for compatibility with SpatialDataBank) ============
        self.max_seq_len = 3000  # Maximum sequence length for gene tokens
        self.mask_ratio = 0.0   # No masking during inference
        self.input_style = "log1p"
        self.validation_split = 0.0  # No validation split for perturbation
        self.accumulation_steps = 1
        
        # Masking and padding tokens
        self.mask_value = -1
        self.pad_value = -2
        self.pad_token = "<pad>"
        self.cls_token = "<cls>"
        
        # Preprocessing parameters
        self.filter_gene_by_counts = False
        self.subset_hvg = False
        self.include_zero_gene = False
        
        # ============ Output Configuration ============
        # Save expression for all steps
        self.save_all_steps = True
        
        # ============ Internal Flags ============
        self.strict_cache_only = True
        self.stratify_by_batch = False

    def to_dict(self):
        """Convert config to dictionary for serialization."""
        def convert(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return str(obj)
        return {k: convert(v) for k, v in self.__dict__.items()}

    def to_json_string(self):
        """Return JSON string representation of config."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


# Alias for backward compatibility
Config = PerturbationConfig
