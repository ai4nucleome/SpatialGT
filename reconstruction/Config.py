"""
SpatialGT Reconstruction Configuration

This module defines configuration classes for reconstruction evaluation.
Supports SpatialGT, SEDR, and KNN baselines.
"""

import json
from pathlib import Path
import torch


class ReconstructionConfig:
    """Configuration class for SpatialGT reconstruction."""
    
    def __init__(self):
        # ============ Path Configuration ============
        self.project_root = Path(__file__).resolve().parent.parent
        
        # Gene embedding files (inherited from pretrain)
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
        
        # ============ Reconstruction Parameters ============
        # Number of iteration steps
        # - SpatialGT: 10 steps (default)
        # - SEDR: 1 step (default)
        # - KNN: 10 steps (default)
        self.steps = 10
        
        # Masking mode: "patch" (BFS from center) or "random"
        self.mask_mode = "patch"
        
        # Number of spots to mask
        self.n_spots = 20
        
        # Initialization mode for masked spots: "zero" (default) or "mean"
        self.mask_init_mode = "zero"
        
        # EMA coefficient (0 = direct replacement)
        self.ema_alpha = 0.0
        
        # ============ Spatial Configuration ============
        self.max_neighbors = 8
        self.distance_threshold = 1800
        self.adjacency_type = "value"
        
        # KNN neighbors for baseline
        self.knn_k = 6
        
        # ============ Data Configuration ============
        self.cache_mode = "h5"
        self.lmdb_path = None
        self.lmdb_manifest_path = None
        self.lmdb_map_size_gb = 1024
        self.lmdb_max_readers = 1024
        
        # ============ Inference Configuration ============
        self.batch_size = 128
        self.num_workers = 4
        self.prefetch_factor = 2
        self.persistent_workers = False
        
        # ============ Metrics Configuration ============
        self.swd_n_projections = 100
        
        # ============ Device & Precision ============
        self.device = "cuda"
        self.dtype = torch.float16
        
        # ============ Reproducibility ============
        self.random_seed = 42
        
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
    
    def save(self, path: str):
        """Save configuration to a JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json_string())


class SEDRConfig:
    """Configuration for SEDR baseline reconstruction."""
    
    def __init__(self):
        # ============ SEDR Training Parameters ============
        self.sedr_epochs = 200
        self.sedr_lr = 0.01
        self.sedr_weight_decay = 0.01
        self.sedr_knn_k = 6
        self.sedr_rec_w = 10.0
        self.sedr_gcn_w = 0.1
        self.sedr_self_w = 1.0
        
        # ============ Reconstruction Parameters ============
        # SEDR default: 1 step (single pass reconstruction)
        self.steps = 1
        self.mask_mode = "patch"
        self.n_spots = 20
        self.ema_alpha = 0.0
        
        # ============ Metrics ============
        self.swd_n_projections = 100
        
        # ============ Device ============
        self.device = "cuda:0"
        
        # ============ Reproducibility ============
        self.random_seed = 42


class KNNConfig:
    """Configuration for KNN baseline reconstruction."""
    
    def __init__(self):
        # ============ KNN Parameters ============
        self.knn_k = 6
        self.inference_mode = "knn_avg"  # "knn_avg" or "ridge"
        self.ridge_alpha = 1.0
        
        # ============ Reconstruction Parameters ============
        # KNN default: 10 steps
        self.steps = 10
        self.mask_mode = "patch"
        self.n_spots = 20
        self.mask_init = "zero"  # "zero" (default) or "mean"
        self.ema_alpha = 0.0
        
        # ============ Metrics ============
        self.swd_n_projections = 100
        
        # ============ Device ============
        self.gpu_id = 0
        
        # ============ Reproducibility ============
        self.random_seed = 42


# Alias for backward compatibility
Config = ReconstructionConfig
