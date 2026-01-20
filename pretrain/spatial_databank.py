import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
import anndata as ad
from anndata import AnnData
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from threading import Lock
from scipy.sparse import issparse, csr_matrix, spmatrix
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import h5py
from tqdm import tqdm
import datetime
import logging
import pickle
import json
import math
import random
from functools import partial
import sys
import warnings
import scipy.sparse as sp
import fnmatch
import traceback
import lmdb

logger = logging.getLogger("SpatialDataBankV2")
from gene_tokenizer import (
    GeneVocab,
    tokenize_and_pad_batch,
    random_mask_value
)

def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits






def safe_highly_variable_genes(adata, **kwargs):
    """
    A wrapper around scanpy's highly_variable_genes that handles the case where 
    bin edges have duplicates.
    """
    try:
        # First attempt: normal call with warnings filtered
        with warnings.catch_warnings():
            # warnings.filterwarnings('ignore', message='No batch_key.*')
            return sc.pp.highly_variable_genes(adata, **kwargs)
    except ValueError as e:
        error_msg = str(e)
        if "Bin edges must be unique" in error_msg:
            logger.warning(f"HVG calculation failed: {error_msg}")
            logger.warning("Attempting to fix duplicate bin edges issue...")
            
            # Monkeypatch pandas.cut to handle duplicates
            original_cut = pd.cut
            
            def patched_cut(*args, **kwargs_cut):
                # Add duplicates='drop' to handle duplicate bin edges
                kwargs_cut['duplicates'] = 'drop'
                return original_cut(*args, **kwargs_cut)
            
            # Apply the monkeypatch
            pd.cut = patched_cut
            
            try:
                # Attempt with monkeypatched pandas.cut
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='No batch_key.*')
                result = sc.pp.highly_variable_genes(adata, **kwargs)
                return result
            except Exception as e2:
                logger.warning(f"Second attempt also failed: {e2}")
                # If n_bins is provided, try reducing it
                if 'n_bins' in kwargs:
                    original_n_bins = kwargs['n_bins']
                    kwargs['n_bins'] = max(10, int(original_n_bins * 0.8))
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message='No batch_key.*')
                        return sc.pp.highly_variable_genes(adata, **kwargs)
                    except Exception as e3:
                        logger.error(f"All attempts failed: {e3}")
                        raise
                else:
                    raise
            finally:
                # Restore original pandas.cut function
                pd.cut = original_cut
        else:
            # Re-raise the original error if it's not related to bin edges
            raise


class Preprocessor:
    """
    Preprocessor for spatial transcriptomics data.
    
    Applies standard preprocessing pipeline:
    1. Filter mitochondrial genes
    2. Filter genes by minimum cells expressing
    3. Filter cells by minimum genes expressed
    4. Library size normalization (normalize_total to target_sum=1e4)
    5. Log1p transformation
    6. Highly variable gene selection
    7. Filter genes by vocabulary
    """
    
    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = True,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        vocab: Optional[GeneVocab] = None,
    ):
        """
        Initialize preprocessor.
        
        Args:
            use_key: Layer key to use as input. None means use adata.X
            filter_gene_by_counts: Minimum counts to keep a gene
            filter_cell_by_counts: Minimum counts to keep a cell
            normalize_total: Target sum for library size normalization
            result_normed_key: Layer key to store normalized data
            log1p: Whether to apply log1p transformation
            result_log1p_key: Layer key to store log1p transformed data
            subset_hvg: Number of highly variable genes to select
            hvg_use_key: Layer to use for HVG calculation
            hvg_flavor: Flavor for HVG calculation ('seurat_v3', 'seurat', 'cell_ranger')
            vocab: Gene vocabulary for filtering
        """
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.vocab = vocab

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> AnnData:
        """
        Apply preprocessing pipeline to AnnData object.
        
        Pipeline:
        1. Remove mitochondrial genes
        2. Filter genes (min_cells=3)
        3. Filter cells (min_genes=10)
        4. Normalize total counts to 1e4
        5. Log1p transform
        6. Select highly variable genes
        7. Filter by vocabulary
        8. Remove zero-expression cells
        
        Args:
            adata: AnnData object to preprocess
            batch_key: Batch key for HVG calculation (optional)
            
        Returns:
            Preprocessed AnnData object
        """
        key_to_process = self.use_key
        if key_to_process == "X":
            key_to_process = None
        
        # Check if data is already log-transformed
        is_logged = self.check_logged(adata, obs_key=key_to_process)
        logger.info(f"Data appears to be log-transformed: {is_logged}")

        # Step 1: Remove mitochondrial genes
        logger.info("Removing mitochondrial genes...")
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        adata = adata[:, ~adata.var['mt']].copy()
        
        # Step 2: Filter genes by minimum cells expressing
        logger.info("Filtering genes (min_cells=3)...")
        sc.pp.filter_genes(adata, min_cells=3)
        
        # Step 3: Filter cells by minimum genes expressed
        logger.info("Filtering cells (min_genes=10)...")
        sc.pp.filter_cells(adata, min_genes=10)
        
        # Step 4: Library size normalization
        logger.info("Normalizing total counts to 1e4...")
        normed_ = sc.pp.normalize_total(
            adata,
            target_sum=1e4,
            layer=key_to_process,
            inplace=False
        )["X"]
        key_to_process = self.result_normed_key or key_to_process
        _set_obs_rep(adata, normed_, layer=key_to_process)

        # Step 5: Log1p transformation
        if self.log1p:
            logger.info("Applying log1p transformation...")
            if is_logged:
                logger.warning(
                    "Input data appears already log1p transformed. "
                    "Set log1p=False to avoid double transformation."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)
        
        # Step 6: Select highly variable genes
        self._process_hvg(adata, batch_key, key_to_process)
        adata = adata[:, adata.var['highly_variable']].copy()
        
        # Step 7: Filter genes by vocabulary
        genes = adata.var_names.tolist()
        
        # Build case-insensitive vocabulary mapping
        vocab_lower_to_idx = {}
        if self.vocab is None:
            raise ValueError("Preprocessor.vocab is None; cannot filter genes by vocabulary")
        try:
            stoi = self.vocab.get_stoi() if hasattr(self.vocab, "get_stoi") else dict(self.vocab)
            for tok, idx in stoi.items():
                low = str(tok).lower()
                if low not in vocab_lower_to_idx:
                    vocab_lower_to_idx[low] = int(idx)
        except Exception as e:
            raise RuntimeError(f"Failed to build vocab mapping: {e}")

        # Match genes with case-insensitive lookup
        genes_lower = [g.lower() for g in genes]
        gene_token_ids = np.array([vocab_lower_to_idx.get(g_low, -1) for g_low in genes_lower], dtype=int)
        valid_genes_mask = gene_token_ids >= 0
        
        n_genes_before = len(genes)
        n_genes_after = sum(valid_genes_mask)
        logger.info(f"Gene filtering: kept {n_genes_after}/{n_genes_before} genes in vocabulary")

        if not all(valid_genes_mask):
            adata = adata[:, valid_genes_mask].copy()
        
        # Step 8: Remove zero-expression cells
        gene_sums = adata.layers["X_log1p"].sum(axis=1)
        if isinstance(gene_sums, np.matrix):
            gene_sums = gene_sums.A1
        cells_to_keep = gene_sums > 0
        if not all(cells_to_keep):
            n_removed = sum(~cells_to_keep)
            logger.info(f"Removed {n_removed} zero-expression cells after preprocessing")
            adata = adata[cells_to_keep].copy()

        # Final check: ensure all required layers exist
        adata = self._ensure_required_layers_exist(adata, key_to_process)

        return adata

    def _process_hvg(self, adata, batch_key, key_to_process):
        """Process highly variable genes selection"""
        if not self.subset_hvg:
            return
            
        logger.info("Subsetting highly variable genes...")
        if batch_key is None:
            logger.warning("No batch_key is provided, will use all cells for HVG selection.")
            
        hvg_params = {
            "layer": self.hvg_use_key,
            "n_top_genes": self.subset_hvg,
            "batch_key": batch_key,
            "flavor": self.hvg_flavor,
            "subset": True,
            "n_bins": 20
        }
        
        # Special settings for seurat_v3 method
        if self.hvg_flavor == "seurat_v3":
            hvg_params["span"] = 0.5
            hvg_params["min_mean"] = 0.01
            
        try:
            safe_highly_variable_genes(adata, **hvg_params)
        except Exception as e:
            logger.warning(f"Failed to calculate HVG with {self.hvg_flavor}: {e}")
            adata = self._try_alternative_hvg_methods(adata, hvg_params)
        return adata


    def _try_alternative_hvg_methods(self, adata, hvg_params):
        """Try alternative HVG calculation methods if the primary method fails"""
        # Try alternative flavors
        alt_flavors = ['seurat', 'cell_ranger', 'dispersion']
        if hvg_params["flavor"] in alt_flavors:
            alt_flavors.remove(hvg_params["flavor"])
        
        success = False
        for alt_flavor in alt_flavors:
            try:
                logger.info(f"Trying alternative flavor: {alt_flavor}")
                # Update parameters
                hvg_params["flavor"] = alt_flavor
                # Special handling for seurat_v3
                if alt_flavor == "seurat_v3":
                    hvg_params["span"] = 0.5
                    hvg_params["min_mean"] = 0.01
                elif alt_flavor == "seurat":
                    hvg_params.pop("span", None)
                
                adata = safe_highly_variable_genes(adata, **hvg_params)
                logger.info(f"Successfully calculated HVG with {alt_flavor}")
                success = True
                break
            except Exception as alt_e:
                logger.warning(f"Alternative flavor {alt_flavor} also failed: {alt_e}")
        
        # Fallback to expression ranking if all methods fail
        if not success:
            adata = self._fallback_to_expression_ranking(adata, hvg_params["n_top_genes"])
        return adata
    def _fallback_to_expression_ranking(self, adata, n_top_genes):
        """Use expression ranking as fallback for HVG selection"""
        logger.warning("All HVG methods failed, using mean expression ranking as fallback")
        # Create pseudo highly variable flag
        adata.var["highly_variable"] = False
        n_genes = min(n_top_genes, adata.n_vars-1)
        
        try:
            mean_expr = adata.X.mean(axis=0)
            if isinstance(mean_expr, np.matrix):
                mean_expr = mean_expr.A1
            elif isinstance(mean_expr, spmatrix):
                mean_expr = mean_expr.toarray().flatten()
            
            top_genes_idx = np.argsort(mean_expr)[-n_genes:]
            adata.var.loc[adata.var.index[top_genes_idx], "highly_variable"] = True
            logger.info(f"Marked {sum(adata.var['highly_variable'])} genes as highly variable")
        except Exception as sort_err:
            logger.error(f"Expression ranking failed: {sort_err}")
            # Last resort - random selection
            random_genes = np.random.choice(adata.n_vars, n_genes, replace=False)
            adata.var["highly_variable"] = False
            adata.var.iloc[random_genes, adata.var.columns.get_loc("highly_variable")] = True
            logger.warning(f"Using {n_genes} randomly selected genes as highly variable")
        return adata
    
    def _ensure_required_layers_exist(self, adata, key_to_process):
        """Ensure all necessary layers exist in the AnnData object"""
        missing_layers = []
        
        # Check which layers are missing
        if hasattr(adata, 'layers'):
            if self.normalize_total and self.result_normed_key not in adata.layers:
                missing_layers.append(self.result_normed_key)
            # if self.log1p and self.result_log1p_key not in adata.layers:
            #     missing_layers.append(self.result_log1p_key)
            
        if missing_layers:
            logger.warning(f"Final check: missing layers: {missing_layers}")
            adata = self._create_missing_layers(adata, missing_layers, key_to_process)
        return adata

    def _create_missing_layers(self, adata, missing_layers, key_to_process):
        """Create any missing layers that should exist"""
        for layer in missing_layers:
            logger.info(f"Attempting to create missing layer: {layer}")
            
            if layer == self.result_normed_key:
                adata = self._create_norm_layer(adata)
            elif layer == self.result_log1p_key:
                adata = self._create_log1p_layer(adata)
        return adata 
    def _create_norm_layer(self, adata):
        """Create normalization layer"""
        try:
            # Try normalization
            sc.pp.normalize_total(adata, target_sum=self.normalize_total, inplace=False)
            adata.layers[self.result_normed_key] = adata.X.copy()
            logger.info(f"Successfully created missing normalization layer")
        except Exception as e:
            logger.error(f"Failed to create normalization layer: {e}")
            # Fallback: use X as substitute
            adata.layers[self.result_normed_key] = adata.X.copy()
        return adata 
    def _create_log1p_layer(self, adata):
        """Create log1p layer"""
        try:
            # Create log1p layer from normalization layer if available
            if self.result_normed_key in adata.layers:
                X_normed = adata.layers[self.result_normed_key]
                if issparse(X_normed):
                    X_normed = X_normed.toarray()
                X_log1p = np.log1p(X_normed)
                adata.layers[self.result_log1p_key] = X_log1p
                logger.info(f"Successfully created missing log1p layer")
            else:
                # Apply log1p directly to X
                orig_X = adata.X.copy()
                sc.pp.log1p(adata)
                adata.layers[self.result_log1p_key] = adata.X.copy()
                adata.X = orig_X
                logger.info(f"Successfully created missing log1p layer")
        except Exception as e:
            logger.error(f"Failed to create log1p layer: {e}")
            # Fallback: use normalization layer or X
            if self.result_normed_key in adata.layers:
                adata.layers[self.result_log1p_key] = adata.layers[self.result_normed_key].copy()
            else:
                adata.layers[self.result_log1p_key] = adata.X.copy()
        return adata 

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """Check if the data is already log1p transformed."""
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True






def _is_all_zeros(X):
    """Check if a matrix contains all zeros"""
    if sp.issparse(X):
        return X.nnz == 0
    else:
        return np.all(X == 0)


def _all_rows_are_zeros(X):
    """Check if all rows in a matrix are zeros"""
    if sp.issparse(X):
        row_sums = np.array(X.sum(axis=1)).flatten()
    else:
        row_sums = np.sum(X, axis=1)
    
    return np.all(row_sums == 0)


class SpatialDataBank:
    """
    Spatial transcriptomics data management system with lazy loading and disk caching.
    """
    
    def __init__(
        self, 
        dataset_paths: List[str], 
        cache_dir: str,
        config: Any,
        truth_file_suffix: str = "_truth.txt",
        force_rebuild: bool = False
    ):
        self.dataset_paths = dataset_paths
        self.config = config
        self.truth_file_suffix = truth_file_suffix
        self.cache_dir = Path(cache_dir)
        self.force_rebuild = force_rebuild
        self.cache_mode = getattr(self.config, "cache_mode", "h5").lower()
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata structure
        self.metadata = {
            "datasets": [],
            "dataset_indices": [],
            "total_spots": 0,
            "total_genes": [],
            "platforms": {},
            "organs": {},
            "species": {},
            "diseases": {}
        }
        # cache helpers for handling sparse global_idx
        self._cached_valid_indices: Optional[List[int]] = None
        self._invalid_global_indices: Set[int] = set()
        
        self._lmdb_env = None
        self._lmdb_env_lock = Lock()
        self.lmdb_path = None
        self.lmdb_manifest = {}
        # Initialize datasets
        if self.cache_mode == "lmdb":
            self._init_lmdb_backend()
        else:
            self._initialize_datasets()

    def _init_lmdb_backend(self):
        """Load metadata and prepare LMDB handles for read-only access"""
        manifest_path = getattr(self.config, "runtime_lmdb_manifest_path", None) or getattr(self.config, "lmdb_manifest_path", None)
        if not manifest_path:
            raise FileNotFoundError("lmdb_manifest_path is not configured. Please set Config.lmdb_manifest_path.")
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"LMDB manifest not found: {manifest_path}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.lmdb_manifest = manifest
        metadata = manifest.get("metadata")
        if metadata:
            self.metadata = metadata
        else:
            self.metadata["datasets"] = manifest.get("datasets", [])
            self.metadata["dataset_indices"] = manifest.get("metadata", {}).get("dataset_indices", [])
            self.metadata["total_spots"] = manifest.get("total_spots", 0)
        self.metadata.setdefault("total_spots", manifest.get("total_spots", 0))
        self.metadata.setdefault("datasets", manifest.get("datasets", []))
        lmdb_path = getattr(self.config, "runtime_lmdb_path", None) or getattr(self.config, "lmdb_path", None)
        if not lmdb_path:
            raise FileNotFoundError("lmdb_path is not configured. Please set Config.lmdb_path.")
        lmdb_path = Path(lmdb_path)
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB file not found: {lmdb_path}")
        self.lmdb_path = lmdb_path
        self._lmdb_env = None
        self._lmdb_env_lock = Lock()
        self.lmdb_env_kwargs = {
            "readonly": True,
            "lock": False,
            "readahead": True,
            "max_readers": getattr(self.config, "lmdb_max_readers", 128),
            "subdir": False,
            "meminit": False,
        }
        logger.info(f"LMDB backend initialized from {self.lmdb_path} (manifest={manifest_path})")
        self._load_skipped_records(manifest)

    def _load_skipped_records(self, manifest: Dict[str, Any]) -> None:
        """Populate invalid index set from manifest skipped_records (if any)."""
        skipped = manifest.get("skipped_records", [])
        if not skipped:
            return
        added = 0
        for rec in skipped:
            idx = rec.get("idx")
            if isinstance(idx, int):
                if idx not in self._invalid_global_indices:
                    self._invalid_global_indices.add(idx)
                    added += 1
        if added > 0:
            logger.info(f"Loaded {added} skipped_records from LMDB manifest; they will be excluded from sampling.")

    def _initialize_datasets(self):
        """Initialize dataset information and create cache index"""
        metadata_file = self.cache_dir / "metadata.json"
        
        # Check for existing cache
        if metadata_file.exists() and not self.force_rebuild:
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded existing metadata with {len(self.metadata['datasets'])} datasets")
                # Normalize each dataset's cache_dir to current config.cache_dir/dataset_name
                changed = False
                for ds in self.metadata.get("datasets", []):
                    ds_name = ds.get("name")
                    if not ds_name:
                        continue
                    desired = str(self.cache_dir / ds_name)
                    if str(ds.get("cache_dir", "")) != desired:
                        ds["cache_dir"] = desired
                        changed = True
                if changed:
                    try:
                        with open(metadata_file, 'w') as f:
                            json.dump(self.metadata, f, indent=2)
                        logger.info("Synchronized dataset cache_dir to current config.cache_dir in metadata.json")
                    except Exception:
                        pass
                return
            except Exception as e:
                if getattr(self.config, 'strict_cache_only', False):
                    raise RuntimeError(
                        f"strict_cache_only=True but failed to read existing metadata: {metadata_file}. Error: {e}"
                    )
                logger.warning(f"Failed to read cached metadata: {e}, will recreate")
        else:
            # metadata.json does not exist
            if getattr(self.config, 'strict_cache_only', False) and not self.force_rebuild:
                raise FileNotFoundError(
                    f"strict_cache_only=True but metadata.json not found: {metadata_file}. "
                    f"Please run preprocessing first, or set strict_cache_only=False to allow cache building during training."
                )
        
        # Initialize metadata collections
        total_genes = set()
        cumulative_count = 0
        skipped_datasets = []
        skipped_reasons = {}
        valid_dataset_paths = []
        
        # Load or initialize platform, organ, species, and disease vocabularies
        # Support disabling vocabulary loading and logging via config.use_metadata_vocabs
        use_vocabs = getattr(self.config, 'use_metadata_vocabs', True)
        log_vocab = getattr(self.config, 'log_vocab_messages', True)
        if use_vocabs:
            all_platforms = self.create_platform_vocabulary()
            platform_count = max(all_platforms.values()) + 1 if all_platforms else 1
            all_organs = self.create_organ_vocabulary()
            organ_count = max(all_organs.values()) + 1 if all_organs else 1
            all_species = self.create_species_vocabulary()
            species_count = max(all_species.values()) + 1 if all_species else 1
            all_diseases = self.create_disease_vocabulary()
            if log_vocab:
                logger.info(f"Initialized platform vocabulary with {len(all_platforms)} items, next ID: {platform_count}")
                logger.info(f"Initialized organ vocabulary with {len(all_organs)} items, next ID: {organ_count}")
                logger.info(f"Initialized species vocabulary with {len(all_species)} items, next ID: {species_count}")
                logger.info(f"Initialized disease vocabulary with {len(all_diseases)} items")
        else:
            all_platforms = {"unknown": 0}
            platform_count = 1
            all_organs = {"unknown": 0}
            organ_count = 1
            all_species = {"unknown": 0}
            species_count = 1
            all_diseases = {"unknown": 0, "healthy": 1, "disease": 2}
        
        print(f"\n🔍 Checking {len(self.dataset_paths)} datasets...")
        
        # Process each dataset path
        for dataset_path in self.dataset_paths:
            try:
                # Determine dataset name and h5ad path
                if dataset_path.endswith('.h5ad'):
                    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
                    h5ad_path = dataset_path
                else:
                    # Search for h5ad files in the dataset_path directory
                    dataset_name = os.path.basename(dataset_path)
                    if os.path.isdir(dataset_path):
                        # If it's a directory, search for h5ad files
                        import glob
                        h5ad_files = glob.glob(os.path.join(dataset_path, "*.h5ad"))
                        if h5ad_files:
                            h5ad_path = h5ad_files[0]  # Use the first h5ad file found
                            dataset_name = os.path.splitext(os.path.basename(h5ad_path))[0]
                        else:
                            h5ad_path = os.path.join(dataset_path, f"{dataset_name}.h5ad")
                    else:
                        # If not a directory, assume it's a file path without extension
                        h5ad_path = f"{dataset_path}.h5ad"
                
                # Check if h5ad file exists
                if not os.path.exists(h5ad_path):
                    reason = f"Processed h5ad file not found: {h5ad_path}"
                    logger.warning(f"⚠️ Dataset {dataset_name}: {reason}")
                    skipped_datasets.append(dataset_name)
                    skipped_reasons[dataset_name] = reason
                    continue
                
                # Try to read file and check for validity
                valid = self._validate_dataset(h5ad_path, dataset_name)
                if not valid[0]:
                    skipped_datasets.append(dataset_name)
                    skipped_reasons[dataset_name] = valid[1]
                    continue
                
                # File exists and is valid
                logger.info(f"✅ Dataset {dataset_name} is valid and will be included")
                print(f"✅ Dataset {dataset_name} is valid and will be included")
                valid_dataset_paths.append((h5ad_path, dataset_name))
                
            except Exception as e:
                # Handle any unexpected errors
                reason = f"Error during processing: {str(e)}"
                logger.error(f"Error checking dataset {dataset_name if 'dataset_name' in locals() else os.path.basename(dataset_path)}: {e}")
                skipped_datasets.append(dataset_name if 'dataset_name' in locals() else os.path.basename(dataset_path))
                skipped_reasons[dataset_name if 'dataset_name' in locals() else os.path.basename(dataset_path)] = reason
        
        # If all datasets were skipped, raise error
        if len(valid_dataset_paths) == 0:
            error_msg = "All datasets were skipped. Ensure valid h5ad files exist in the data_clean directory."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Output summary information
        self._print_dataset_summary(valid_dataset_paths, skipped_datasets, skipped_reasons)
        
        print("🔄 Preprocessing datasets and building metadata...")
        
        # Process valid dataset paths
        for dataset_idx, (dataset_path, dataset_name) in enumerate(valid_dataset_paths):
            try:
                # Create dataset cache directory
                dataset_cache_dir = self.cache_dir / dataset_name
                dataset_cache_dir.mkdir(exist_ok=True)
                
                # Process the dataset
                dataset_info = self._process_dataset(
                    dataset_idx, 
                    dataset_path, 
                    dataset_name,
                    dataset_cache_dir,
                    all_platforms,
                    platform_count,
                    all_organs,
                    organ_count,
                    all_species,
                    species_count,
                    all_diseases,
                    total_genes,
                    cumulative_count
                )
                
                if dataset_info:
                    # Update cumulative count
                    cumulative_count = dataset_info["end_idx"]
                    # Update platform counter if needed
                    if "platform_id" in dataset_info and dataset_info["platform_id"] >= platform_count:
                        platform_count = dataset_info["platform_id"] + 1
                    # Update organ counter if needed
                    if "organ_id" in dataset_info and dataset_info["organ_id"] >= organ_count:
                        organ_count = dataset_info["organ_id"] + 1
                    # Update species counter if needed
                    if "species_id" in dataset_info and dataset_info["species_id"] >= species_count:
                        species_count = dataset_info["species_id"] + 1
                
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
                traceback.print_exc()
                continue
        
        # Update metadata
        self.metadata["total_genes"] = list(total_genes)
        self.metadata["total_spots"] = cumulative_count
        self.metadata["platforms"] = all_platforms
        self.metadata["organs"] = all_organs
        self.metadata["species"] = all_species
        self.metadata["diseases"] = all_diseases
        
        print(f"Initialized {len(self.metadata['datasets'])} datasets with {cumulative_count} valid spots, {len(total_genes)} total genes")
        if getattr(self.config, 'log_vocab_messages', False):
            print(f"Found {len(all_platforms)} different platforms, {len(all_organs)} organ types, {len(all_diseases)} disease states, {len(all_species)} species types")
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _validate_dataset(self, h5ad_path, dataset_name):
        """Validate a dataset for inclusion"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_adata = ad.read_h5ad(h5ad_path)
            
            # Check if matrix is all zeros
            if _is_all_zeros(test_adata.X):
                reason = f"Dataset matrix contains all zero values"
                logger.warning(f"⚠️ Dataset {dataset_name}: {reason}")
                return False, reason
            
            # Check matrix dimensions
            if test_adata.shape[0] == 0 or test_adata.shape[1] == 0:
                reason = f"Dataset matrix has zero dimensions (shape: {test_adata.shape})"
                logger.warning(f"⚠️ Dataset {dataset_name}: {reason}")
                return False, reason
            
            # Check if all rows are zeros
            if _all_rows_are_zeros(test_adata.X):
                reason = f"All rows in dataset are zero values"
                logger.warning(f"⚠️ Dataset {dataset_name}: {reason}")
                return False, reason
                
            # Check if spatial coordinates exist
            has_spatial = False
            if hasattr(test_adata, 'obsm') and 'spatial' in test_adata.obsm:
                has_spatial = True
            elif hasattr(test_adata, 'obs') and all(c in test_adata.obs.columns for c in ['x', 'y']):
                has_spatial = True
            
            if not has_spatial:
                reason = f"Dataset lacks spatial coordinate information"
                logger.warning(f"⚠️ Dataset {dataset_name}: {reason}")
                return False, reason
            
            # Check if dataset has too many spots (>1,000,000)
            if test_adata.n_obs > 1000000:
                reason = f"Dataset has too many spots ({test_adata.n_obs:,} > 1,000,000)"
                logger.warning(f"⚠️ Dataset {dataset_name}: {reason}")
                return False, reason

            # Check if dataset has enough spots for neighbor calculation
            max_neighbors = getattr(self.config, 'max_neighbors', 0)
            if max_neighbors > 0 and test_adata.n_obs <= max_neighbors:
                reason = f"Dataset has too few spots ({test_adata.n_obs}) for max_neighbors={max_neighbors}"
                logger.warning(f"⚠️ Dataset {dataset_name}: {reason}")
                return False, reason
            
            del test_adata  # Release memory
            return True, ""
            
        except Exception as e:
            reason = f"Error during reading: {str(e)}"
            logger.warning(f"⚠️ Dataset {dataset_name}: {reason}")
            return False, reason

    def _print_dataset_summary(self, valid_paths, skipped_datasets, skipped_reasons):
        """Print summary of dataset processing results"""
        if skipped_datasets:
            print("\n📊 Dataset check results summary:")
            print(f"⚠️ Total of {len(skipped_datasets)} datasets were skipped:")
            for dataset_name in skipped_datasets:
                print(f"   - {dataset_name}: {skipped_reasons.get(dataset_name, 'Unknown reason')}")
            print(f"✅ Remaining {len(valid_paths)} valid datasets will be used for analysis:")
            for i, (path, name) in enumerate(valid_paths):
                print(f"   {i+1}. {name}")
            print("\n")
        else:
            print(f"\n✅ All {len(self.dataset_paths)} datasets are valid and will be used for analysis\n")

    def _process_dataset(self, dataset_idx, dataset_path, dataset_name, 
                         dataset_cache_dir, all_platforms, platform_count,
                         all_organs, organ_count, all_species, species_count, all_diseases,
                         total_genes, cumulative_count):
        """Process a single dataset and return its information"""
        # Check for preprocessed cache
        processed_file = dataset_cache_dir / "processed.h5ad"
        
        # Initialize variables
        adata = None
        original_shape = None
        
        # Try to load from cache if available
        if processed_file.exists() and not self.force_rebuild:
            try:
                adata = ad.read_h5ad(processed_file)
                adata.var_names_make_unique()
                logger.info(f"Read preprocessed data from cache: {processed_file}")
                
                # Check if the cached data is valid
                if adata.shape[0] == 0 or adata.shape[1] == 0 or _all_rows_are_zeros(adata.X):
                    logger.warning(f"Cached dataset {dataset_name} is invalid (shape: {adata.shape}), will skip")
                    print(f"⚠️ Dataset {dataset_name} is invalid (shape: {adata.shape}), skipping")
                    return None
            except Exception as e:
                logger.warning(f"Failed to read preprocessing cache: {e}, will reprocess")
                adata = None
        
        # If not successfully loaded from cache, read from original h5ad
        if adata is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    adata = ad.read_h5ad(dataset_path)
                    adata.var_names_make_unique()
                    original_shape = adata.shape
                    
                    # Check if the original data is valid
                    if adata.shape[0] == 0 or adata.shape[1] == 0 or _all_rows_are_zeros(adata.X):
                        logger.warning(f"Dataset {dataset_name} is invalid (shape: {adata.shape}), will skip")
                        print(f"⚠️ Dataset {dataset_name} is invalid (shape: {adata.shape}), skipping")
                        return None
            except Exception as e:
                logger.error(f"Error reading dataset {dataset_name}: {e}")
                print(f"⚠️ Dataset {dataset_name} reading failed: {str(e)}, skipping")
                return None
        
        # Process metadata information
        metadata_info, adata = self._process_metadata(
            adata, dataset_name, all_platforms, platform_count, 
            all_organs, organ_count, all_species, species_count, all_diseases
        )
        
        platform_info = metadata_info["platform"]
        platform_id = metadata_info["platform_id"]
        organ_info = metadata_info["organ"]
        organ_id = metadata_info["organ_id"]
        disease_info = metadata_info["disease"]
        disease_id = metadata_info["disease_id"]
        species_info = metadata_info["species"]
        species_id = metadata_info["species_id"]    
        
        # Update platform dictionary if needed
        if platform_info.lower() not in all_platforms and platform_info.lower() != "unknown":
            # Assign a new ID for this platform (incremental)
            new_platform_id = platform_count
            all_platforms[platform_info.lower()] = new_platform_id
            platform_count += 1
            platform_id = new_platform_id  # Update the platform_id for this dataset
            logger.info(f"Assigned new platform ID {new_platform_id} to '{platform_info}'")
            
            # Save updated platform vocabulary
            if hasattr(self.config, 'platform_vocab_path') and self.config.platform_vocab_path:
                try:
                    with open(self.config.platform_vocab_path, 'w') as f:
                        json.dump(all_platforms, f, indent=2)
                    logger.info(f"Updated platform vocabulary saved to {self.config.platform_vocab_path}")
                except Exception as e:
                    logger.warning(f"Failed to save updated platform vocabulary: {e}")
        
        # Update organ dictionary if needed
        if organ_info.lower() not in all_organs and organ_info.lower() != "unknown":
            # Assign a new ID for this organ (incremental)
            new_organ_id = organ_count
            all_organs[organ_info.lower()] = new_organ_id
            organ_count += 1
            organ_id = new_organ_id  # Update the organ_id for this dataset
            logger.info(f"Assigned new organ ID {new_organ_id} to '{organ_info}'")
            
            # Save updated organ vocabulary
            if hasattr(self.config, 'organ_vocab_path') and self.config.organ_vocab_path:
                try:
                    with open(self.config.organ_vocab_path, 'w') as f:
                        json.dump(all_organs, f, indent=2)
                    logger.info(f"Updated organ vocabulary saved to {self.config.organ_vocab_path}")
                except Exception as e:
                    logger.warning(f"Failed to save updated organ vocabulary: {e}")
        
        # Update species dictionary if needed
        if species_info.lower() not in all_species and species_info.lower() != "unknown":
            # Assign a new ID for this species (incremental)
            new_species_id = species_count
            all_species[species_info.lower()] = new_species_id
            species_count += 1
            species_id = new_species_id  # Update the species_id for this dataset
            logger.info(f"Assigned new species ID {new_species_id} to '{species_info}'")
            
            # Save updated species vocabulary
            if hasattr(self.config, 'species_vocab_path') and self.config.species_vocab_path:
                try:
                    with open(self.config.species_vocab_path, 'w') as f:
                        json.dump(all_species, f, indent=2)
                    logger.info(f"Updated species vocabulary saved to {self.config.species_vocab_path}")
                except Exception as e:
                    logger.warning(f"Failed to save updated species vocabulary: {e}")
        
        # Ensure required columns exist
        adata = self._ensure_required_columns(adata, dataset_name)
        
        # Preprocess the data
        adata = self._preprocess_adata(adata, dataset_idx)
        
        # If preprocessing returned None, skip this dataset
        if adata is None:
            logger.warning(f"Dataset {dataset_name} preprocessing resulted in invalid data, skipping")
            print(f"⚠️ Dataset {dataset_name} preprocessing resulted in invalid data, skipping")
            return None
            
        # Save to cache
        try:
            adata.write_h5ad(processed_file)
            logger.info(f"Preprocessed data saved to: {processed_file}")
            
            # Show filtering results
            if original_shape and original_shape[0] > adata.shape[0]:
                logger.info(f"Before filtering: {original_shape[0]} spots, after: {adata.shape[0]} spots")
                print(f"⚠️ Dataset {dataset_name}: Filtered out {original_shape[0] - adata.shape[0]} all-zero expression rows")
        except Exception as save_error:
            logger.error(f"Error saving preprocessed dataset {dataset_name}: {save_error}")
            print(f"⚠️ Dataset {dataset_name} save failed: {str(save_error)}")
        
        # Get gene list
        genes_list = self._get_gene_list(adata)
        
        # Check if cell type labels exist
        has_labels = hasattr(adata, 'obs') and "celltype" in adata.obs
        
        # Build dataset info, including platform, organ, and disease info
        info = {
            "n_spots": adata.n_obs,
            "n_genes": adata.n_vars,
            "genes": genes_list,
            "has_labels": has_labels,
            "platform": platform_info,
            "platform_id": platform_id,
            "organ": organ_info,
            "organ_id": organ_id,
            "species": species_info,
            "species_id": species_id,
            "disease": disease_info,
            "disease_id": disease_id
        }
        
        # Save info to cache
        info_cache = dataset_cache_dir / "info.json"
        with open(info_cache, 'w') as f:
            json.dump(info, f, indent=2)
        
        # Collect all genes
        total_genes.update(genes_list)
        
        # Record dataset position in global index
        start_idx = cumulative_count
        end_idx = start_idx + info["n_spots"]
        
        # Add dataset info
        self.metadata["datasets"].append({
            "name": dataset_name,
            "path": dataset_path,
            "cache_dir": str(dataset_cache_dir),
            "n_spots": info["n_spots"],
            "n_genes": info["n_genes"],
            "has_labels": info["has_labels"],
            "platform": platform_info,
            "platform_id": platform_id,
            "organ": organ_info,
            "organ_id": organ_id,
            "disease": disease_info,
            "disease_id": disease_id,
            "species": species_info,
            "species_id": species_id
        })
        
        self.metadata["dataset_indices"].append({
            "dataset_idx": dataset_idx,
            "start_idx": start_idx,
            "end_idx": end_idx
        })
        
        # Update global indices if needed
        self._update_global_indices(adata, start_idx, end_idx, processed_file)
        
        # Return information for this dataset
        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "platform_id": platform_id,
            "organ_id": organ_id,
            "disease_id": disease_id,
            "species_id": species_id
        }

    def _process_metadata(self, adata, dataset_name, all_platforms, platform_count, 
                          all_organs, organ_count, all_species, species_count, all_diseases):
        """Process metadata information from AnnData"""
        # Use the global vocabularies (all_platforms, all_organs, all_species, all_diseases)
        # These are either loaded from existing files or created during initialization
        # We'll match against these vocabularies and update them if needed
        self.platform_vocab = all_platforms
        self.organ_vocab = all_organs
        self.disease_vocab = all_diseases
        self.species_vocab = all_species
        # Get platform information
        platform_info = self._get_metadata_field(adata, 'platform', "unknown")
        
        # Case-insensitive platform matching with predefined vocabulary
        platform_key = platform_info.lower()
        platform_id = None
        
        # Try direct lookup first
        if platform_key in self.platform_vocab:
            platform_id = self.platform_vocab[platform_key]
        else:
            # Try case-insensitive matching
            for vocab_key in self.platform_vocab:
                if vocab_key.lower() == platform_key:
                    platform_id = self.platform_vocab[vocab_key]
                    break
        
        # If no match found, use unknown
        if platform_id is None:
            platform_id = self.platform_vocab.get("unknown", 0)
            
        # For backward compatibility, also update all_platforms
        # But don't assign IDs here - that happens in _process_dataset
        
        # Add platform info to adata.uns
        adata.uns['platform'] = platform_info
        
        # Get organ information
        organ_info = self._get_metadata_field(adata, 'organ', "unknown")
        
        # Case-insensitive organ matching with predefined vocabulary
        organ_key = organ_info.lower()
        organ_id = None
        
        # Try direct lookup first
        if organ_key in self.organ_vocab:
            organ_id = self.organ_vocab[organ_key]
        else:
            # Try case-insensitive matching
            for vocab_key in self.organ_vocab:
                if vocab_key.lower() == organ_key:
                    organ_id = self.organ_vocab[vocab_key]
                    break
        
        # If no match found, use unknown
        if organ_id is None:
            organ_id = self.organ_vocab.get("unknown", 0)
            
        # For backward compatibility, also update all_organs
        # But don't assign IDs here - that happens in _process_dataset
        
        # Add organ info to adata.uns
        adata.uns['organ'] = organ_info
        
        # Get species information
        species_info = self._get_metadata_field(adata, 'species', "unknown")
        
        # Case-insensitive species matching with predefined vocabulary
        species_key = species_info.lower()
        species_id = None
        
        # Try direct lookup first
        if species_key in self.species_vocab:
            species_id = self.species_vocab[species_key]
        else:
            # Try case-insensitive matching
            for vocab_key in self.species_vocab:
                if vocab_key.lower() == species_key:
                    species_id = self.species_vocab[vocab_key]
                    break
        
        # If no match found, use unknown
        if species_id is None:
            species_id = self.species_vocab.get("unknown", 0)
            
        # For backward compatibility, also update all_species
        # But don't assign IDs here - that happens in _process_dataset
        
        # Add species info to adata.uns
        adata.uns['species'] = species_info
        
        # Get disease information
        disease_info = self._get_metadata_field(adata, 'disease', "unknown")
        
        # Standardize disease info
        if disease_info.lower() in ["control", "healthy", "normal"]:
            disease_info = "healthy"
        elif disease_info.lower() != "unknown":
            disease_info = "disease"
        
        # Get disease ID from predefined vocabulary
        disease_id = self.disease_vocab.get(disease_info.lower(), self.disease_vocab.get("unknown", 0))
        
        # Add disease info to adata.uns
        adata.uns['disease'] = disease_info
        
        if getattr(self.config, 'log_vocab_messages', False):
            print(f"Dataset {dataset_name}: platform='{platform_info}' (ID={platform_id}), organ='{organ_info}' (ID={organ_id}), species='{species_info}' (ID={species_id}), disease='{disease_info}' (ID={disease_id})")
        
        return {
            "platform": platform_info,
            "platform_id": platform_id,
            "organ": organ_info,
            "organ_id": organ_id,
            "disease": disease_info,
            "disease_id": disease_id,
            "species": species_info,
            "species_id": species_id
        }, adata

    def _get_metadata_field(self, adata, field_name, default_value):
        """Get a metadata field from AnnData.uns, with validation"""
        field_value = default_value
        
        if hasattr(adata, 'uns') and field_name in adata.uns:
            field_data = adata.uns[field_name]
            
            # Standardize field data format
            if isinstance(field_data, bytes):
                field_value = field_data.decode('utf-8')
            elif isinstance(field_data, str) and field_data.strip():
                field_value = field_data.strip()
            elif isinstance(field_data, list) and field_data:
                # If non-empty list, use first non-empty element
                for item in field_data:
                    if item and str(item).strip():
                        field_value = str(item).strip()
                        break
            elif isinstance(field_data, dict) and field_data:
                # If non-empty dict, use first key
                field_value = str(next(iter(field_data))).strip()
            else:
                # Other cases, try to convert to string
                temp_value = str(field_data).strip()
                field_value = temp_value if temp_value else default_value
        
        # If value is empty or invalid, use default
        if not field_value or field_value.lower() in ["none", "null", ""]:
            field_value = default_value
            
        return field_value

    def _ensure_required_columns(self, adata, dataset_name):
        """Ensure required columns exist in AnnData"""
        # Add gene_name if it doesn't exist
        if hasattr(adata, 'var') and "gene_name" not in adata.var:
            try:
                adata.var["gene_name"] = adata.var.index.tolist()
                logger.info(f"Created gene_name column for dataset {dataset_name}")
            except Exception as e:
                logger.warning(f"Failed to create gene_name column for dataset {dataset_name}: {e}")
        
        # Ensure batch_id exists
        if hasattr(adata, 'obs') and "batch_id" not in adata.obs:
            adata.obs["batch_id"] = adata.obs["str_batch"] = dataset_name
            logger.info(f"Created batch_id column for dataset {dataset_name}")
        
        # Ensure celltype info exists
        if hasattr(adata, 'obs') and "celltype" not in adata.obs:
            adata.obs["celltype"] = "unknown"
            logger.info(f"Dataset {dataset_name} has no cell type info, using 'unknown'")
        
        # Ensure celltype_id exists
        if hasattr(adata, 'obs') and "celltype_id" not in adata.obs:
            if hasattr(adata, 'obs') and "celltype" in adata.obs:
                # If labels exist, create ID
                try:
                    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
                    adata.obs["celltype_id"] = celltype_id_labels
                    logger.info(f"Created celltype_id column for dataset {dataset_name}")
                except Exception as e:
                    logger.warning(f"Failed to create celltype_id column for dataset {dataset_name}: {e}")
                    adata.obs["celltype_id"] = -1
            else:
                # If no labels, set to -1
                adata.obs["celltype_id"] = -1
                logger.info(f"Dataset {dataset_name} has no cell type info, celltype_id set to -1")
        return adata
    def _minimal_processing(self, adata, processed_file):
        """Apply minimal processing when regular preprocessing fails"""
        # At least save counts layer
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        
        # At least filter cells with all zeros
        sc.pp.filter_cells(adata, min_counts=1)
        
        # Save basic processed data
        adata.write_h5ad(processed_file)
        logger.info(f"Saved basic filtered data to: {processed_file}")

    def _get_gene_list(self, adata):
        """Get gene list from AnnData"""
        if hasattr(adata, 'var') and "gene_name" in adata.var:
            genes_list = adata.var["gene_name"].tolist()
        else:
            genes_list = adata.var_names.tolist()
        return genes_list

    def _update_global_indices(self, adata, start_idx, end_idx, processed_file):
        """Update global indices if needed"""
        update_global_idx = False
        
        if "global_idx" not in adata.obs:
            update_global_idx = True
        else:
            # Check if existing global index matches current start_idx
            current_start = adata.obs["global_idx"].iloc[0] if len(adata.obs["global_idx"]) > 0 else None
            if current_start != start_idx:
                update_global_idx = True
        
        if update_global_idx:
            # Update global index
            adata.obs["global_idx"] = range(start_idx, end_idx)
            # Save updated adata
            adata.write_h5ad(processed_file)
            logger.info(f"Updated global indices and saved data: {processed_file}")

    def _preprocess_adata(self, adata: ad.AnnData, dataset_idx: int) -> Optional[ad.AnnData]:
        """
        Preprocess AnnData object with log-normalization pipeline.
        
        Args:
            adata: AnnData object to preprocess
            dataset_idx: Dataset index for logging
            
        Returns:
            Preprocessed AnnData or None if preprocessing fails
        """
        dataset_name = self.metadata["datasets"][dataset_idx]["name"] if dataset_idx < len(self.metadata["datasets"]) else "unknown"
        
        # Check if dataset is empty or invalid before preprocessing
        if adata.shape[0] == 0 or adata.shape[1] == 0 or _all_rows_are_zeros(adata.X):
            logger.warning(f"Dataset {dataset_name} is invalid before preprocessing (shape: {adata.shape}), skipping")
            return None
            
        # Save original counts
        if hasattr(adata, 'layers') and "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()

        # Create preprocessor with log-normalization pipeline
        vocab = self.prepare_vocabulary()
        preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=self.config.filter_gene_by_counts,
            filter_cell_by_counts=1,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,  # Always use log1p transformation
            result_log1p_key="X_log1p",
            subset_hvg=self.config.subset_hvg,
            hvg_flavor="seurat_v3",
            vocab=vocab
        )
        
        try:
            adata = preprocessor(adata, batch_key=None)
            
            # Validate preprocessing result
            if adata.shape[0] == 0 or adata.shape[1] == 0 or _all_rows_are_zeros(adata.X):
                logger.warning(f"Dataset {dataset_name} became invalid after preprocessing (shape: {adata.shape}), skipping")
                return None
            
        except Exception as e:
            logger.error(f"Error during preprocessing dataset {dataset_name}: {e}")
            return None
            
        return adata
    

    def _handle_preprocessing_error(self, adata, preprocessor, error):
        """Handle errors during preprocessing"""
        error_msg = str(error)
        logger.warning(f"Error during preprocessing: {error}")
        
        # Handle numerical issues in LOESS fitting
        if "reciprocal condition number" in error_msg or "singularities" in error_msg:
            logger.warning(f"Numerical issues while calculating highly variable genes")
            
            # Perform basic processing steps
            self._perform_basic_processing(adata, preprocessor)
            
            # Try alternative HVG calculation methods
            if preprocessor.subset_hvg:
                self._try_alternative_hvg_calculation(adata, preprocessor)
            

    def _perform_basic_processing(self, adata, preprocessor):
        """Perform basic processing steps"""
        # Filter zero expression rows
        if preprocessor.filter_cell_by_counts:
            logger.info("Filtering zero expression rows...")
            sc.pp.filter_cells(adata, min_counts=1)
        
        # Filter low expression genes
        if preprocessor.filter_gene_by_counts:
            logger.info("Filtering low expression genes...")
            min_counts = 3 if isinstance(preprocessor.filter_gene_by_counts, bool) else preprocessor.filter_gene_by_counts
            sc.pp.filter_genes(adata, min_counts=min_counts)
        
        # Normalization and log transformation
        if preprocessor.normalize_total:
            logger.info("Performing normalization...")
            sc.pp.normalize_total(adata, target_sum=preprocessor.normalize_total)
            adata.layers[preprocessor.result_normed_key] = adata.X.copy()
        
        if preprocessor.log1p:
            logger.info("Performing log transformation...")
            sc.pp.log1p(adata)
            adata.layers[preprocessor.result_log1p_key] = adata.X.copy()

    def _try_alternative_hvg_calculation(self, adata, preprocessor):
        """Try different methods to calculate highly variable genes"""
        methods = ['seurat', 'cell_ranger', 'dispersion']
        success = False
        
        for method in methods:
            try:
                logger.info(f"Trying '{method}' method for HVG calculation...")
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='No batch_key.*')
                    sc.pp.highly_variable_genes(
                        adata, 
                        flavor=method,
                        n_top_genes=min(2000, adata.n_vars - 1),
                        batch_key=None
                    )
                adata.var["highly_variable"] = adata.var["highly_variable"].fillna(False)
                logger.info(f"Successfully calculated HVG using '{method}' method")
                success = True
                break
            except Exception as hvg_err:
                logger.warning(f"Failed with '{method}' method: {hvg_err}")
        
        # If all methods fail, use expression ranking as fallback
        if not success:
            self._use_expression_ranking_fallback(adata)

    def _use_expression_ranking_fallback(self, adata):
        """Use expression ranking as fallback for HVG selection"""
        logger.warning("All HVG calculation methods failed, using expression ranking as fallback")
        # Create pseudo highly variable gene marking
        adata.var["highly_variable"] = False
        try:
            # Select top 2000 genes by mean expression
            mean_expr = adata.X.mean(axis=0)
            if isinstance(mean_expr, np.matrix):
                mean_expr = mean_expr.A1
            elif isinstance(mean_expr, spmatrix):
                mean_expr = mean_expr.toarray().flatten()
            
            top_genes_idx = np.argsort(mean_expr)[-min(2000, adata.n_vars-1):]
            adata.var.loc[adata.var.index[top_genes_idx], "highly_variable"] = True
            logger.info(f"Marked {sum(adata.var['highly_variable'])} genes as highly variable")
        except Exception as sort_err:
            logger.error(f"Expression ranking failed: {sort_err}")
            # Last resort - random selection
            n_genes = min(2000, adata.n_vars-1)
            random_genes = np.random.choice(adata.n_vars, n_genes, replace=False)
            adata.var["highly_variable"] = False
            adata.var.iloc[random_genes, adata.var.columns.get_loc("highly_variable")] = True
            logger.warning(f"Using {n_genes} randomly selected genes as highly variable")


    def create_platform_vocabulary(self):
        """
        Load platform vocabulary from predefined file or create and save a new one if loading fails
        
        Returns:
            platform_vocab: Dictionary mapping platform names to IDs
        """
        # If metadata vocabularies are disabled, return minimal vocabulary without logging
        if not getattr(self.config, 'use_metadata_vocabs', True):
            return {"unknown": 0}
        # Create default platform vocabulary - always start with unknown
        platform_vocab = {"unknown": 0}
        
        # If no path is provided in config, create a default path in cache_dir
        if not hasattr(self.config, 'platform_vocab_path') or self.config.platform_vocab_path is None:
            default_vocab_dir = os.path.join(self.cache_dir, "vocabularies")
            os.makedirs(default_vocab_dir, exist_ok=True)
            default_vocab_path = os.path.join(default_vocab_dir, "platform_vocab.json")
            if getattr(self.config, 'log_vocab_messages', False):
                logger.info(f"No platform_vocab_path provided in config, using default path: {default_vocab_path}")
            
            # Save the vocabulary to the default path
            try:
                with open(default_vocab_path, 'w') as f:
                    json.dump(platform_vocab, f, indent=2)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Saved platform vocabulary to default path: {default_vocab_path}")
                # Update config with the default path
                self.config.platform_vocab_path = default_vocab_path
            except Exception as e:
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.warning(f"Failed to save platform vocabulary to default path: {e}")
                
            return platform_vocab
            
        # Load from predefined vocabulary file
        predefined_vocab_path = self.config.platform_vocab_path
        
        # Try loading from predefined file
        try:
            with open(predefined_vocab_path, 'r') as f:
                platform_vocab = json.load(f)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Loaded platform vocabulary from predefined file, size: {len(platform_vocab)}")
                return platform_vocab
        except Exception as e:
            if getattr(self.config, 'log_vocab_messages', False):
                logger.warning(f"Failed to read predefined platform vocabulary: {e}, will create default")
                logger.warning(f"Using minimal platform vocabulary with only 'unknown' key")
            
            # Save the newly created vocabulary
            try:
                os.makedirs(os.path.dirname(predefined_vocab_path), exist_ok=True)
                with open(predefined_vocab_path, 'w') as f:
                    json.dump(platform_vocab, f, indent=2)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Saved newly created platform vocabulary to {predefined_vocab_path}")
            except Exception as save_err:
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.warning(f"Failed to save platform vocabulary: {save_err}")
                
            return platform_vocab

    def create_organ_vocabulary(self):
        """
        Load organ vocabulary from predefined file or create and save a new one if loading fails
        
        Returns:
            organ_vocab: Dictionary mapping organ names to IDs
        """
        if not getattr(self.config, 'use_metadata_vocabs', True):
            return {"unknown": 0}
        # Create default organ vocabulary - always start with unknown
        organ_vocab = {"unknown": 0}
        
        # If no path is provided in config, create a default path in cache_dir
        if not hasattr(self.config, 'organ_vocab_path') or self.config.organ_vocab_path is None:
            default_vocab_dir = os.path.join(self.cache_dir, "vocabularies")
            os.makedirs(default_vocab_dir, exist_ok=True)
            default_vocab_path = os.path.join(default_vocab_dir, "organ_vocab.json")
            if getattr(self.config, 'log_vocab_messages', False):
                logger.info(f"No organ_vocab_path provided in config, using default path: {default_vocab_path}")
            
            # Save the vocabulary to the default path
            try:
                with open(default_vocab_path, 'w') as f:
                    json.dump(organ_vocab, f, indent=2)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Saved organ vocabulary to default path: {default_vocab_path}")
                # Update config with the default path
                self.config.organ_vocab_path = default_vocab_path
            except Exception as e:
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.warning(f"Failed to save organ vocabulary to default path: {e}")
                
            return organ_vocab
            
        # Load from predefined vocabulary file
        predefined_vocab_path = self.config.organ_vocab_path
        
        # Try loading from predefined file
        try:
            with open(predefined_vocab_path, 'r') as f:
                organ_vocab = json.load(f)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Loaded organ vocabulary from predefined file, size: {len(organ_vocab)}")
                return organ_vocab
        except Exception as e:
            if getattr(self.config, 'log_vocab_messages', False):
                logger.warning(f"Failed to read predefined organ vocabulary: {e}, will create default")
                logger.warning(f"Using minimal organ vocabulary with only 'unknown' key")
            
            # Save the newly created vocabulary
            try:
                os.makedirs(os.path.dirname(predefined_vocab_path), exist_ok=True)
                with open(predefined_vocab_path, 'w') as f:
                    json.dump(organ_vocab, f, indent=2)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Saved newly created organ vocabulary to {predefined_vocab_path}")
            except Exception as save_err:
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.warning(f"Failed to save organ vocabulary: {save_err}")
                
            return organ_vocab
    
    def create_species_vocabulary(self):
        """
        Load species vocabulary from predefined file or create and save a new one if loading fails
        
        Returns:
            species_vocab: Dictionary mapping species names to IDs
        """
        if not getattr(self.config, 'use_metadata_vocabs', True):
            return {"unknown": 0}
        # Create default species vocabulary - always start with unknown
        species_vocab = {"unknown": 0}
        
        # If no path is provided in config, create a default path in cache_dir
        if not hasattr(self.config, 'species_vocab_path') or self.config.species_vocab_path is None:
            default_vocab_dir = os.path.join(self.cache_dir, "vocabularies")
            os.makedirs(default_vocab_dir, exist_ok=True)
            default_vocab_path = os.path.join(default_vocab_dir, "species_vocab.json")
            if getattr(self.config, 'log_vocab_messages', False):
                logger.info(f"No species_vocab_path provided in config, using default path: {default_vocab_path}")
            
            # Save the vocabulary to the default path
            try:
                with open(default_vocab_path, 'w') as f:
                    json.dump(species_vocab, f, indent=2)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Saved species vocabulary to default path: {default_vocab_path}")
                # Update config with the default path
                self.config.species_vocab_path = default_vocab_path
            except Exception as e:
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.warning(f"Failed to save species vocabulary to default path: {e}")
                
            return species_vocab
            
        # Load from predefined vocabulary file
        predefined_vocab_path = self.config.species_vocab_path
        
        # Try loading from predefined file
        try:
            with open(predefined_vocab_path, 'r') as f:
                species_vocab = json.load(f)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Loaded species vocabulary from predefined file, size: {len(species_vocab)}")
                return species_vocab
        except Exception as e:
            if getattr(self.config, 'log_vocab_messages', False):
                logger.warning(f"Failed to read predefined species vocabulary: {e}, will create default")
                logger.warning(f"Using minimal species vocabulary with only 'unknown' key")
            
            # Save the newly created vocabulary
            try:
                os.makedirs(os.path.dirname(predefined_vocab_path), exist_ok=True)
                with open(predefined_vocab_path, 'w') as f:
                    json.dump(species_vocab, f, indent=2)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Saved newly created species vocabulary to {predefined_vocab_path}")
            except Exception as save_err:
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.warning(f"Failed to save species vocabulary: {save_err}")
                
            return species_vocab
    
    def create_disease_vocabulary(self):
        """
        Load disease vocabulary from predefined file or create and save a new one if loading fails
        
        Returns:
            disease_vocab: Dictionary mapping disease states to IDs
        """
        if not getattr(self.config, 'use_metadata_vocabs', True):
            return {"unknown": 0, "healthy": 1, "disease": 2}
        # Create default disease vocabulary - always start with unknown, healthy, disease
        disease_vocab = {"unknown": 0, "healthy": 1, "disease": 2}
        
        # If no path is provided in config, create a default path in cache_dir
        if not hasattr(self.config, 'disease_vocab_path') or self.config.disease_vocab_path is None:
            default_vocab_dir = os.path.join(self.cache_dir, "vocabularies")
            os.makedirs(default_vocab_dir, exist_ok=True)
            default_vocab_path = os.path.join(default_vocab_dir, "disease_vocab.json")
            if getattr(self.config, 'log_vocab_messages', False):
                logger.info(f"No disease_vocab_path provided in config, using default path: {default_vocab_path}")
            
            # Save the vocabulary to the default path
            try:
                with open(default_vocab_path, 'w') as f:
                    json.dump(disease_vocab, f, indent=2)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Saved disease vocabulary to default path: {default_vocab_path}")
                # Update config with the default path
                self.config.disease_vocab_path = default_vocab_path
            except Exception as e:
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.warning(f"Failed to save disease vocabulary to default path: {e}")
                
            return disease_vocab
            
        # Load from predefined vocabulary file
        predefined_vocab_path = self.config.disease_vocab_path
        
        # Try loading from predefined file
        try:
            with open(predefined_vocab_path, 'r') as f:
                disease_vocab = json.load(f)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Loaded disease vocabulary from predefined file, size: {len(disease_vocab)}")
                return disease_vocab
        except Exception as e:
            if getattr(self.config, 'log_vocab_messages', False):
                logger.warning(f"Failed to read predefined disease vocabulary: {e}, will create default")
                logger.warning(f"Using default disease vocabulary: {list(disease_vocab.keys())}")
            
            # Save the newly created vocabulary
            try:
                os.makedirs(os.path.dirname(predefined_vocab_path), exist_ok=True)
                with open(predefined_vocab_path, 'w') as f:
                    json.dump(disease_vocab, f, indent=2)
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.info(f"Saved newly created disease vocabulary to {predefined_vocab_path}")
            except Exception as save_err:
                if getattr(self.config, 'log_vocab_messages', False):
                    logger.warning(f"Failed to save disease vocabulary: {save_err}")
                
            return disease_vocab

    def _get_dataset_by_idx(self, dataset_idx: int) -> Tuple[ad.AnnData, Dict]:
        """
        Get dataset by index
        
        Args:
            dataset_idx: Dataset index
            
        Returns:
            adata: AnnData object
            info: Dataset information
        """
        dataset_info = self.metadata["datasets"][dataset_idx]
        dataset_name = dataset_info["name"]
        
        # Read from preprocessed cache
        processed_file = Path(dataset_info["cache_dir"]) / "processed.h5ad"
        
        try:
            # Since preprocessing is done during initialization, just read the cache
            adata = ad.read_h5ad(processed_file)
            logger.info(f"Read dataset: {dataset_name}, shape: {adata.shape}")
            
            # Verify dataset is not all zeros
            if adata.shape[0] > 0 and adata.shape[1] > 0:
                if _is_all_zeros(adata.X):
                    error_msg = f"Dataset {dataset_name} contains all zero values, skipping"
                    logger.warning(error_msg)
                    raise ValueError(f"ZeroValueDataset: {error_msg}")
                
                # Check if all rows are zeros
                if _all_rows_are_zeros(adata.X):
                    error_msg = f"Dataset {dataset_name} has all rows as zero values, skipping"
                    logger.warning(error_msg)
                    raise ValueError(f"ZeroValueDataset: {error_msg}")
            else:
                error_msg = f"Dataset {dataset_name} has invalid shape: {adata.shape}, skipping"
                logger.warning(error_msg)
                raise ValueError(f"ZeroValueDataset: {error_msg}")
            
            return adata, dataset_info
        except Exception as e:
            error_msg = str(e)
            if "ZeroValueDataset" in error_msg:
                # Re-raise zero value dataset exception for special handling
                raise ValueError(error_msg)
            else:
                error_msg = f"Failed to read dataset {dataset_name}: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def get_neighbors_for_spot(self, global_idx: int):
        """
        Get neighbors for a specific spot - searches only within its dataset
        
        Args:
            global_idx: Global spot index
            
        Returns:
            List of neighbor indices or empty list if none
        """
        if self.cache_mode == "lmdb":
            record = self._get_spot_from_lmdb(global_idx)
            return list(record.get("neighbor_indices", []))
        # 1. Determine which dataset the spot belongs to
        dataset_idx = None
        
        for info in self.metadata["dataset_indices"]:
            if info["start_idx"] <= global_idx < info["end_idx"]:
                dataset_idx = info["dataset_idx"]
                break
        
        if dataset_idx is None:
            logger.warning(f"Global index {global_idx} not found in any known dataset")
            return []
        
        # 2. Lazy load neighbors index dictionary
        if not hasattr(self, '_neighbors_index'):
            neighbors_dir = self.cache_dir / "neighbors"
            index_file = neighbors_dir / "neighbors_index.json"
            
            if not index_file.exists():
                print("Neighbors index file doesn't exist. Please follow instructions in preprocess.py to create neighbor index files.")
                return []
            else:
                with open(index_file, 'r') as f:
                    raw_index = json.load(f)
                # Remap to current cache_dir/neighbors; fallback to glob by dataset name
                remapped = {}
                try:
                    import glob as _glob, re as _re
                    for idx_str, old_path in raw_index.items():
                        final_path = None
                        try:
                            idx_int = int(idx_str)
                        except Exception:
                            remapped[idx_str] = old_path
                            continue
                        ds_name = None
                        if 0 <= idx_int < len(self.metadata.get("datasets", [])):
                            ds_name = self.metadata["datasets"][idx_int].get("name")
                        if isinstance(old_path, str):
                            candidate = neighbors_dir / os.path.basename(old_path)
                            if candidate.exists():
                                final_path = str(candidate)
                        if final_path is None and ds_name:
                            pattern = str(neighbors_dir / f"neighbors_{ds_name}_n*_v3.h5")
                            matches = sorted(_glob.glob(pattern))
                            if matches:
                                prefer = None
                                desired_k = int(getattr(self.config, 'max_neighbors', 0))
                                for m in matches:
                                    base = os.path.basename(m)
                                    m_k = _re.search(r"_n(\d+)_v3\.h5$", base)
                                    if m_k and int(m_k.group(1)) == desired_k:
                                        prefer = m
                                        break
                                final_path = prefer or matches[0]
                        remapped[idx_str] = final_path or old_path
                except Exception:
                    remapped = raw_index
                self._neighbors_index = remapped
                # Try write back
                try:
                    with open(index_file, 'w') as f:
                        json.dump(self._neighbors_index, f, indent=2)
                except Exception:
                    pass
        
        # 3. Lazy load dataset's neighbors file
        if not hasattr(self, '_neighbors_files_dict'):
            self._neighbors_files_dict = {}
        
        # Get neighbors file for current dataset
        str_dataset_idx = str(dataset_idx)
        if str_dataset_idx not in self._neighbors_index:
            logger.warning(f"Neighbors file for dataset index {dataset_idx} not found")
            return []
        
        # Open file if not already open
        if str_dataset_idx not in self._neighbors_files_dict:
            neighbors_file_path = self._neighbors_index[str_dataset_idx]
            # If path not absolute or missing, rebuild under current cache_dir
            try:
                if isinstance(neighbors_file_path, str) and not os.path.isabs(neighbors_file_path):
                    neighbors_file_path = str((self.cache_dir / "neighbors" / os.path.basename(neighbors_file_path)))
            except Exception:
                pass
            if not isinstance(neighbors_file_path, str) or not os.path.exists(neighbors_file_path):
                # Last resort: glob by dataset name
                try:
                    ds_name = self.metadata["datasets"][dataset_idx].get("name")
                    import glob as _glob
                    pattern = str((self.cache_dir / "neighbors" / f"neighbors_{ds_name}_n*_v3.h5"))
                    matches = _glob.glob(pattern)
                    if matches:
                        neighbors_file_path = matches[0]
                except Exception:
                    pass
            try:
                self._neighbors_files_dict[str_dataset_idx] = h5py.File(neighbors_file_path, 'r')
                logger.debug(f"Opened neighbors file for dataset {dataset_idx}")
            except Exception as e:
                logger.error(f"Failed to open neighbors file for dataset {dataset_idx}: {e}")
                return []
        
        # 4. Get neighbors from file
        h5_file = self._neighbors_files_dict[str_dataset_idx]
        
        try:
            # Check if spot has neighbor data
            neighbors_group = h5_file['neighbors']
            spot_key = str(global_idx)
            
            if spot_key not in neighbors_group:
                logger.warning(f"No neighbor data found for spot {global_idx}")
                return []
            
            # Read and filter padding values
            neighbors_data = neighbors_group[spot_key][:]
            valid_neighbors = [n for n in neighbors_data if n >= 0]
            
            return valid_neighbors
        
        except Exception as e:
            logger.error(f"Failed to read neighbor data for spot {global_idx}: {e}")
            return []

    def _get_all_valid_indices(self) -> List[int]:
        """
        Build (and cache) the list of global indices that have backing data.

        The metadata may contain sparse ranges (holes), so we always derive the
        candidate indices from the recorded dataset ranges instead of assuming
        0..total_spots-1.
        """
        if self._cached_valid_indices is not None:
            return self._cached_valid_indices

        indices: List[int] = []
        dataset_indices = self.metadata.get("dataset_indices", [])
        for info in dataset_indices:
            start_idx = info.get("start_idx")
            end_idx = info.get("end_idx")
            if isinstance(start_idx, int) and isinstance(end_idx, int) and end_idx > start_idx:
                indices.extend(range(start_idx, end_idx))

        self._cached_valid_indices = indices
        return indices

    def mark_invalid_global_idx(self, global_idx: int) -> None:
        """Record a missing global index so later sampling can avoid it."""
        if global_idx is None:
            return
        self._invalid_global_indices.add(int(global_idx))

    def is_global_idx_invalid(self, global_idx: int) -> bool:
        """Check whether a global index was previously marked as missing."""
        return int(global_idx) in self._invalid_global_indices

    def _filter_invalid_indices(self, indices: List[int], label: str) -> List[int]:
        """Drop indices that are already known to be invalid."""
        if not self._invalid_global_indices:
            return indices
        filtered = [idx for idx in indices if idx not in self._invalid_global_indices]
        removed = len(indices) - len(filtered)
        if removed > 0:
            logger.warning(
                f"{label} split: filtered {removed} indices missing in LMDB (kept {len(filtered)})"
            )
        return filtered

    def get_data_split(self, validation_split=0.1, stratify_by_batch=True,  split="train", random_seed=42):
        """
        Get train/validation split
        
        Args:
            validation_split: Validation set proportion
            stratify_by_batch: Whether to stratify by batch
            random_seed: Random seed
            
        Returns:
            train_indices, val_indices: Training and validation indices
        """
        # Enumerate all candidate indices from metadata (may be sparse)
        all_indices = self._get_all_valid_indices()

        if split == "all" or validation_split <= 0:
            return self._filter_invalid_indices(all_indices, "all"), []
        
        split_cache = self.cache_dir / f"split_v{validation_split}_s{stratify_by_batch}_r{random_seed}.json"
        if split_cache.exists() and not self.force_rebuild:
            try:
                with open(split_cache, 'r') as f:
                    splits = json.load(f)
                train_cached = self._filter_invalid_indices(splits.get("train", []), "train(cache)")
                val_cached = self._filter_invalid_indices(splits.get("val", []), "val(cache)")
                return train_cached, val_cached
            except Exception as e:
                logger.warning(f"Failed to read dataset split cache: {e}, will recreate")
        
        if stratify_by_batch:
            # Stratified sampling by batch to maintain proportions
            train_indices = []
            val_indices = []
            
            for info in self.metadata["dataset_indices"]:
                start_idx = info.get("start_idx")
                end_idx = info.get("end_idx")
                if start_idx is None or end_idx is None or end_idx <= start_idx:
                    continue
                batch_indices = list(range(start_idx, end_idx))
                
                # Split for each batch separately
                np.random.seed(random_seed)
                np.random.shuffle(batch_indices)
                split_idx = int(len(batch_indices) * (1 - validation_split))
                
                train_indices.extend(batch_indices[:split_idx])
                val_indices.extend(batch_indices[split_idx:])
        else:
            # Simple random split
            np.random.seed(random_seed)
            np.random.shuffle(all_indices)
            split_idx = int(len(all_indices) * (1 - validation_split))
            
            train_indices = all_indices[:split_idx]
            val_indices = all_indices[split_idx:]
        
        # Save to cache
        with open(split_cache, 'w') as f:
            json.dump({"train": train_indices, "val": val_indices}, f)
        
        train_indices = self._filter_invalid_indices(train_indices, "train")
        val_indices = self._filter_invalid_indices(val_indices, "val")
        return train_indices, val_indices

    def preprocess_dataset_spots(self, dataset_idx):
        """Preprocess all spots for a single dataset"""
        # Get dataset info
        dataset_info = self.metadata["datasets"][dataset_idx]
        dataset_name = dataset_info["name"]
        
        # Ensure dataset_indices exists in metadata
        if "dataset_indices" not in self.metadata or dataset_idx >= len(self.metadata["dataset_indices"]):
            logger.error(f"Dataset {dataset_name} indices not found in metadata")
            return None
            
        start_idx = self.metadata["dataset_indices"][dataset_idx]["start_idx"]
        end_idx = self.metadata["dataset_indices"][dataset_idx]["end_idx"]
        n_spots = end_idx - start_idx
        
        # Check if spot count is 0
        if n_spots <= 0:
            logger.warning(f"Dataset {dataset_name} contains no valid spots, skipping")
            return None
            
        # Create HDF5 file path
        spots_file = Path(dataset_info["cache_dir"]) / "spots.h5"
        # Check if already exists
        if spots_file.exists() and not self.force_rebuild:
            logger.info(f"Spot data for dataset {dataset_name} already exists: {spots_file}")
            return spots_file
        
        # Load dataset (using already preprocessed AnnData)
        logger.info(f"Creating spot data file for dataset {dataset_name}...")
        try:
            adata, _ = self._get_dataset_by_idx(dataset_idx)
            # Check if data is empty or all zeros
            if adata.shape[0] == 0 or adata.shape[1] == 0:
                logger.warning(f"Dataset {dataset_name} has zero dimensions: {adata.shape}, skipping")
                return None
                
            if _is_all_zeros(adata.X):
                logger.warning(f"Dataset {dataset_name} contains all zero values, skipping")
                return None
                
        except ValueError as e:
            if "ZeroValueDataset" in str(e):
                logger.warning(f"Dataset {dataset_name} contains all zero values, skipping: {e}")
                return None
            else:
                logger.error(f"Failed to read dataset {dataset_name}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading dataset {dataset_name}: {e}")
            return None

        # Get metadata information
        metadata_info = self._get_spot_metadata(dataset_info, adata)
        platform_info = metadata_info["platform"] 
        platform_id = metadata_info["platform_id"]
        organ_info = metadata_info["organ"]
        organ_id = metadata_info["organ_id"]
        species_info = metadata_info["species"]
        species_id = metadata_info["species_id"]
        disease_info = metadata_info["disease"]
        disease_id = metadata_info["disease_id"]
        
        if getattr(self.config, 'log_vocab_messages', False):
            logger.info(f"Dataset {dataset_name} platform: {platform_info} (ID: {platform_id})")
            logger.info(f"Dataset {dataset_name} organ: {organ_info} (ID: {organ_id})")
            logger.info(f"Dataset {dataset_name} disease: {disease_info} (ID: {disease_id})")
            logger.info(f"Dataset {dataset_name} species: {species_info} (ID: {species_id})")
        # Ensure required layers exist
        self._ensure_spot_data_layers(adata, dataset_name)
        
        # Choose expression layer based on input style
        layer_key = {
            "normed_raw": "X_normed",
            "log1p": "X_log1p",
        }[self.config.input_style]
        
        # Get gene list
        genes = self._get_gene_list_for_spots(adata, dataset_name)
        
        # Get vocabulary
        vocab = self.prepare_vocabulary()
        # Create HDF5 file
        with h5py.File(spots_file, 'w') as f:
            # Set file attributes
            f.attrs['dataset_name'] = dataset_name
            f.attrs['n_spots'] = n_spots
            f.attrs['input_style'] = self.config.input_style
            f.attrs['include_zero_gene'] = int(self.config.include_zero_gene)
            
            # Add platform, organ and disease information
            f.attrs['platform'] = platform_info  
            f.attrs['platform_id'] = platform_id
            f.attrs['organ'] = organ_info
            f.attrs['organ_id'] = organ_id
            f.attrs['disease'] = disease_info
            f.attrs['disease_id'] = disease_id
            f.attrs['species'] = species_info
            f.attrs['species_id'] = species_id
            # Store gene names
            gene_names_ascii = np.array([g.encode('ascii', 'ignore') for g in genes])
            f.create_dataset('gene_names', data=gene_names_ascii)
            
            # Create spots group
            spots_group = f.create_group('spots')
            
            # Process each spot with progress display
            try:
                from tqdm import tqdm
                iterator = tqdm(range(n_spots), desc=f"Processing spots for {dataset_name}")
            except ImportError:
                iterator = range(n_spots)
            
            for local_idx in iterator:
                global_idx = start_idx + local_idx
                self._process_single_spot(
                    adata, spots_group, global_idx, local_idx, 
                    layer_key, genes, vocab, dataset_name,
                    platform_info, platform_id,
                    organ_info, organ_id,
                    disease_info, disease_id,
                    species_info, species_id
                )
        
        # Return created file path
        logger.info(f"Spot data for dataset {dataset_name} saved to {spots_file}")
        print(f"Spot data for dataset {dataset_name} saved to {spots_file}")
        return spots_file

    def _get_spot_metadata(self, dataset_info, adata):
        """Get metadata for spot processing"""
        # Always create new vocabulary instances for each dataset
        # This ensures each vocabulary type is independent and not shared
        self.platform_vocab = self.create_platform_vocabulary()
        self.organ_vocab = self.create_organ_vocabulary()
        self.disease_vocab = self.create_disease_vocabulary()
        self.species_vocab = self.create_species_vocabulary()
        # Get platform info - first try from dataset_info
        if "platform" in dataset_info:
            platform_info = dataset_info["platform"]
        else:
            platform_info = adata.uns.get('platform', "unknown") if hasattr(adata, 'uns') else "unknown"
        
        # Ensure platform info is valid
        if not platform_info or platform_info.lower() in ["none", "null", ""]:
            platform_info = "unknown"
        
        # Case-insensitive platform matching
        platform_key = platform_info.lower()
        platform_id = None
        
        # Try direct lookup first
        if platform_key in self.platform_vocab:
            platform_id = self.platform_vocab[platform_key]
        else:
            # Try case-insensitive matching
            for vocab_key in self.platform_vocab:
                if vocab_key.lower() == platform_key:
                    platform_id = self.platform_vocab[vocab_key]
                    break
        
        # If no match found, use unknown
        if platform_id is None:
            platform_id = self.platform_vocab.get("unknown", 0)
        
        # Get organ info - first try from dataset_info
        if "organ" in dataset_info:
            organ_info = dataset_info["organ"]
        else:
            organ_info = adata.uns.get('organ', "unknown") if hasattr(adata, 'uns') else "unknown"
        
        # Ensure organ info is valid
        if not organ_info or organ_info.lower() in ["none", "null", ""]:
            organ_info = "unknown"
        
        # Case-insensitive organ matching
        organ_key = organ_info.lower()
        organ_id = None
        
        # Try direct lookup first
        if organ_key in self.organ_vocab:
            organ_id = self.organ_vocab[organ_key]
        else:
            # Try case-insensitive matching
            for vocab_key in self.organ_vocab:
                if vocab_key.lower() == organ_key:
                    organ_id = self.organ_vocab[vocab_key]
                    break
        
        # If no match found, use unknown
        if organ_id is None:
            organ_id = self.organ_vocab.get("unknown", 0)
        
        # Get species info - first try from dataset_info
        if "species" in dataset_info:
            species_info = dataset_info["species"]
        else:
            species_info = adata.uns.get('species', "unknown") if hasattr(adata, 'uns') else "unknown"
        
        # Ensure species info is valid
        if not species_info or species_info.lower() in ["none", "null", ""]:
            species_info = "unknown"
        
        # Case-insensitive species matching
        species_key = species_info.lower()
        species_id = None
        
        # Try direct lookup first
        if species_key in self.species_vocab:
            species_id = self.species_vocab[species_key]
        else:
            # Try case-insensitive matching
            for vocab_key in self.species_vocab:
                if vocab_key.lower() == species_key:
                    species_id = self.species_vocab[vocab_key]
                    break
        
        # If no match found, use unknown
        if species_id is None:
            species_id = self.species_vocab.get("unknown", 0)
        
        # Get disease info - first try from dataset_info
        if "disease" in dataset_info:
            disease_info = dataset_info["disease"]
        else:
            disease_info = adata.uns.get('disease', "unknown") if hasattr(adata, 'uns') else "unknown"
        
        # Ensure disease info is valid
        if not disease_info or disease_info.lower() in ["none", "null", ""]:
            disease_info = "unknown"
        
        # Standardize disease info
        if disease_info.lower() in ["control", "healthy", "normal"]:
            disease_info = "healthy"
        elif disease_info.lower() != "unknown":
            disease_info = "disease"
        
        # Get disease ID
        disease_id = self.disease_vocab.get(disease_info.lower(), self.disease_vocab.get("unknown", 0))
        
        # Get species info - first try from dataset_info
        if "species" in dataset_info:
            species_info = dataset_info["species"]
        else:
            species_info = adata.uns.get('species', "unknown") if hasattr(adata, 'uns') else "unknown"
        
        return {
            "platform": platform_info,
            "platform_id": platform_id,
            "organ": organ_info,
            "organ_id": organ_id,
            "disease": disease_info,
            "disease_id": disease_id,
            "species": species_info,
            "species_id": species_id
        }

    def _ensure_spot_data_layers(self, adata, dataset_name):
        """Ensure all required layers exist for spot data processing"""
        # 1. Create normalization layer if missing
        if hasattr(adata, 'layers') and "X_normed" not in adata.layers and adata.X is not None:
            logger.warning(f"Creating normalization layer for dataset {dataset_name}...")
            try:
                orig_X = adata.X.copy()
                sc.pp.normalize_total(adata, target_sum=1e4)
                adata.layers["X_normed"] = adata.X.copy()
                adata.X = orig_X
                logger.info(f"Successfully created normalization layer")
            except Exception as e:
                logger.error(f"Failed to create normalization layer: {e}")
                adata.layers["X_normed"] = adata.X.copy()

        # 2. Create log1p layer if missing
        if hasattr(adata, 'layers') and "X_log1p" not in adata.layers:
            logger.warning(f"Creating log1p layer for dataset {dataset_name}...")
            try:
                X_normed = adata.layers["X_normed"]
                if issparse(X_normed):
                    X_normed = X_normed.toarray()
                adata.layers["X_log1p"] = np.log1p(X_normed)
                logger.info(f"Successfully created log1p layer")
            except Exception as e:
                logger.error(f"Failed to create log1p layer: {e}")
                adata.layers["X_log1p"] = adata.layers["X_normed"].copy()
        

    def _get_gene_list_for_spots(self, adata, dataset_name):
        """Get gene list for spot data processing"""
        try:
            if "gene_name" in adata.var:
                genes = adata.var["gene_name"].tolist()
                logger.info(f"Got {len(genes)} gene names from adata.var['gene_name']")
            else:
                genes = adata.var_names.tolist()
                logger.info(f"Got {len(genes)} gene names from adata.var_names")
                
                # If first time processing, add gene_name column
                try:
                    adata.var["gene_name"] = adata.var_names
                    logger.info(f"Created adata.var['gene_name'] column")
                except Exception as e:
                    logger.warning(f"Failed to create gene_name column: {e}")
        except Exception as e:
            logger.error(f"Failed to get gene names: {e}, using index as substitute")
            # Use numeric index as gene names
            genes = [f"gene_{i}" for i in range(adata.n_vars)]
            
        return genes



    def _process_single_spot(self, adata, spots_group, global_idx, local_idx, 
                           layer_key, genes, vocab, dataset_name,
                           platform_info, platform_id, 
                           organ_info, organ_id,
                           disease_info, disease_id,
                           species_info, species_id):
        """Process a single spot's data"""
        # Create spot subgroup
        spot_group = spots_group.create_group(str(global_idx))
        
        # Get expression data safely
        try:
            # Use log1p data for normal preprocessing
            if issparse(adata.layers["X_log1p"]):
                raw_normed = adata.layers["X_log1p"][local_idx].toarray().flatten()
            else:
                raw_normed = adata.layers["X_log1p"][local_idx].flatten()
            
            # Ensure data is 1D and has proper data types for HDF5 compatibility
            try:
                # Ensure data is 1D
                if hasattr(raw_normed, 'flatten'):
                    raw_normed = raw_normed.flatten()
                    
                # Convert to proper data types
                raw_normed = np.asarray(raw_normed, dtype=np.float32)
                
                # Ensure array has the same length as adata.n_vars
                if len(raw_normed) != adata.n_vars:
                    logger.warning(f"Raw normalized data length mismatch for spot {global_idx}: expected {adata.n_vars}, got {len(raw_normed)}")
                    if len(raw_normed) > adata.n_vars:
                        raw_normed = raw_normed[:adata.n_vars]
                    else:
                        # Pad with zeros if too short
                        padding = np.zeros(adata.n_vars - len(raw_normed), dtype=np.float32)
                        raw_normed = np.concatenate([raw_normed, padding])
                        
            except Exception as conv_err:
                logger.error(f"Data conversion error for spot {global_idx}: {conv_err}")
                logger.error(f"raw_normed type: {type(raw_normed)}")
                raise
  
        except Exception as expr_err:
            logger.error(f"Failed to get expression data for spot {global_idx}: {expr_err}")
            # Use zero vector as fallback
            raw_normed = np.zeros(adata.n_vars, dtype=np.float32)

        # Save raw normalized expression data (compressed)
        spot_group.create_dataset('raw_normed', data=raw_normed, compression="gzip", compression_opts=4)
        # Get cell type and batch info safely
        try:
            celltype = str(adata.obs["celltype"].iloc[local_idx]) if "celltype" in adata.obs else "unknown"
            celltype_id = int(adata.obs["celltype_id"].iloc[local_idx]) if "celltype_id" in adata.obs else -1
            batch_id = str(adata.obs["batch_id"].iloc[local_idx]) if "batch_id" in adata.obs else dataset_name
        except Exception as e:
            logger.warning(f"Failed to get metadata for spot {global_idx}: {e}, using defaults")
            celltype = "unknown"
            celltype_id = -1
            batch_id = dataset_name
        
        # Save metadata
        spot_group.attrs['celltype'] = celltype
        spot_group.attrs['celltype_id'] = celltype_id
        spot_group.attrs['batch_id'] = batch_id
        
        # Save platform, organ and disease info
        spot_group.attrs['platform'] = platform_info
        spot_group.attrs['platform_id'] = platform_id
        spot_group.attrs['organ'] = organ_info
        spot_group.attrs['organ_id'] = organ_id
        spot_group.attrs['disease'] = disease_info
        spot_group.attrs['disease_id'] = disease_id
        spot_group.attrs['species'] = species_info
        spot_group.attrs['species_id'] = species_id
        # Save spatial coordinates
        try:
            spot_group.create_dataset('spatial_coords', data=adata.obsm["spatial"][local_idx])
        except Exception as e:
            logger.warning(f"Failed to save spatial coordinates for spot {global_idx}: {e}, using defaults")
            # Use default coordinates [0, 0] as fallback
            spot_group.create_dataset('spatial_coords', data=np.zeros(2))
        # Preprocessing: tokenization and zero expression filtering
        # Load vocabulary from config
        vocab_path = self.config.vocab_file
        vocab_lower_to_idx = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        for tok, idx in vocab_json.items():
            low = tok.lower()
            if low not in vocab_lower_to_idx:
                vocab_lower_to_idx[low] = idx
        vocab = vocab_lower_to_idx
        
        # Convert gene names to lowercase for matching
        genes = [gene.lower() for gene in genes]
        
        gene_token_ids = np.array([vocab[gene] if gene in vocab else -1 for gene in genes])
        valid_mask = gene_token_ids >= 0  # Filter genes not in vocabulary
        gene_token_ids = gene_token_ids[valid_mask]
        raw_normed_filtered = raw_normed[valid_mask]
        spot_group.create_dataset('gene_ids', data=gene_token_ids, compression="gzip", compression_opts=4)

        spot_group.create_dataset('raw_normed_values', data=raw_normed_filtered, compression="gzip", compression_opts=4)
      
        # Add statistics as attributes
        spot_group.attrs['n_genes'] = len(gene_token_ids)
        spot_group.attrs['n_expressed_genes'] = np.sum(raw_normed > 0)

    def get_spot_data(self, global_idx: int, include_raw: bool = False):
        """
        Efficiently get preprocessed data for a single spot
        
        Args:
            global_idx: Global spot index
            include_raw: Whether to include raw expression data
            
        Returns:
            Spot data dictionary
        """
        if self.cache_mode == "lmdb":
            return self._get_spot_from_lmdb(global_idx)
        # Find corresponding dataset
        dataset_idx = None
        for info in self.metadata["dataset_indices"]:
            if info["start_idx"] <= global_idx < info["end_idx"]:
                dataset_idx = info["dataset_idx"]
                break
        
        if dataset_idx is None:
            raise ValueError(f"Global index {global_idx} not found in any dataset")
        
        # Lazy load dataset file dictionary
        if not hasattr(self, '_dataset_files_dict'):
            self._dataset_files_dict = {}
            self._dataset_open_files = {}
        
        # Get dataset info (prefer current config.cache_dir as root for dataset cache)
        dataset_info = self.metadata["datasets"][dataset_idx]
        dataset_name = dataset_info.get("name")
        dataset_cache_dir = Path(self.cache_dir) / dataset_name if dataset_name else Path(dataset_info.get("cache_dir", self.cache_dir))
        spots_file = dataset_cache_dir / "spots.h5"
        # If metadata cache_dir differs from current config, update in-place and persist
        if str(dataset_info.get("cache_dir", "")) != str(dataset_cache_dir):
            dataset_info["cache_dir"] = str(dataset_cache_dir)
            try:
                metadata_file = Path(self.cache_dir) / "metadata.json"
                if metadata_file.exists():
                    import json as _json
                    with open(metadata_file, 'w') as f:
                        _json.dump(self.metadata, f, indent=2)
            except Exception:
                pass
        
        # Always create new vocabulary instances
        # This ensures each vocabulary type is independent and not shared
        self.platform_vocab = self.create_platform_vocabulary()
        self.organ_vocab = self.create_organ_vocabulary()
        self.disease_vocab = self.create_disease_vocabulary()
        self.species_vocab = self.create_species_vocabulary()
        # Check if file exists
        if not spots_file.exists():
            logger.error(f"File doesn't exist: {spots_file}, absolute path: {spots_file.absolute()}")
            logger.error(f"Dataset info: {dataset_info}")
            raise FileNotFoundError(f"Spot data file not found for dataset {dataset_name}: {spots_file}")
        
        # Use file handle cache pool
        if dataset_idx not in self._dataset_open_files:
            # First access, open file
            try:
                h5file = h5py.File(spots_file, 'r')
                self._dataset_open_files[dataset_idx] = h5file
            except Exception as e:
                logger.error(f"Failed to open spot data file: {e}")
                raise
        
        # Get file handle
        h5file = self._dataset_open_files[dataset_idx]
        
        # Get platform info - first try from file attributes
        platform_name = "unknown"
        platform_id = 0  # Default ID 0 for "unknown"
        
        try:
            if 'platform' in h5file.attrs:
                platform_name = h5file.attrs['platform']
                if isinstance(platform_name, bytes):
                    platform_name = platform_name.decode('utf-8')
            
            if 'platform_id' in h5file.attrs:
                platform_id = int(h5file.attrs['platform_id'])
            else:
                # Case-insensitive platform matching
                platform_key = platform_name.lower()
                platform_id = None
                
                # Try direct lookup first
                if platform_key in self.platform_vocab:
                    platform_id = self.platform_vocab[platform_key]
                else:
                    # Try case-insensitive matching
                    for vocab_key in self.platform_vocab:
                        if vocab_key.lower() == platform_key:
                            platform_id = self.platform_vocab[vocab_key]
                            break
                
                # If no match found, use unknown
                if platform_id is None:
                    platform_id = self.platform_vocab.get("unknown", 0)
                
            # If platform name is invalid, use "unknown"
            if not platform_name or platform_name.lower() in ["none", "null", ""]:
                platform_name = "unknown"
                platform_id = 0
                
        except Exception as e:
            if getattr(self.config, 'log_vocab_messages', False):
                logger.warning(f"Failed to get platform info: {e}, using default 'unknown'")
            platform_name = "unknown"
            platform_id = 0

        # Get organ info - first try from file attributes
        organ_name = "unknown"
        organ_id = 0  # Default ID 0 for "unknown"
        
        try:
            if 'organ' in h5file.attrs:
                organ_name = h5file.attrs['organ']
                if isinstance(organ_name, bytes):
                    organ_name = organ_name.decode('utf-8')
            
            if 'organ_id' in h5file.attrs:
                organ_id = int(h5file.attrs['organ_id'])
            else:
                # Case-insensitive organ matching
                organ_key = organ_name.lower()
                organ_id = None
                
                # Try direct lookup first
                if organ_key in self.organ_vocab:
                    organ_id = self.organ_vocab[organ_key]
                else:
                    # Try case-insensitive matching
                    for vocab_key in self.organ_vocab:
                        if vocab_key.lower() == organ_key:
                            organ_id = self.organ_vocab[vocab_key]
                            break
                
                # If no match found, use unknown
                if organ_id is None:
                    organ_id = self.organ_vocab.get("unknown", 0)
                
            # If organ name is invalid, use "unknown"
            if not organ_name or organ_name.lower() in ["none", "null", ""]:
                organ_name = "unknown"
                organ_id = 0

            # Get species info - first try from file attributes
            species_name = "unknown"
            species_id = 0  # Default ID 0 for "unknown"
            
            if 'species' in h5file.attrs:
                species_name = h5file.attrs['species']
                if isinstance(species_name, bytes):
                    species_name = species_name.decode('utf-8')
            
            if 'species_id' in h5file.attrs:
                species_id = int(h5file.attrs['species_id'])
            else:
                # Case-insensitive species matching
                species_key = species_name.lower()
                species_id = None
                
                # Try direct lookup first
                if species_key in self.species_vocab:
                    species_id = self.species_vocab[species_key]
                else:
                    # Try case-insensitive matching
                    for vocab_key in self.species_vocab:
                        if vocab_key.lower() == species_key:
                            species_id = self.species_vocab[vocab_key]
                            break
                
                # If no match found, use unknown
                if species_id is None:
                    species_id = self.species_vocab.get("unknown", 0)
                
            # If species name is invalid, use "unknown"
            if not species_name or species_name.lower() in ["none", "null", ""]:
                species_name = "unknown"
                species_id = 0

        except Exception as e:
            if getattr(self.config, 'log_vocab_messages', False):
                logger.warning(f"Failed to get organ info: {e}, using default 'unknown'")
            organ_name = "unknown"
            organ_id = 0
            species_name = "unknown"
            species_id = 0
        
        # Access spot data
        try:
            spot_group = h5file['spots'][str(global_idx)]
            # First check if required fields exist
            has_raw_normed_values = 'raw_normed_values' in spot_group
            if not has_raw_normed_values:
                raise ValueError(f"Spot {global_idx} lacks raw_normed_values")

            # Basic data
            spot_data = {
                "global_idx": global_idx,
                "gene_ids": spot_group['gene_ids'][:],
                "raw_normed_values": spot_group['raw_normed_values'][:],
                "celltype": spot_group.attrs['celltype'],
                "celltype_id": int(spot_group.attrs['celltype_id']),
                "batch_id": spot_group.attrs['batch_id'],
                "platform": platform_name,
                "platform_id": platform_id,
                "organ": organ_name,
                "organ_id":organ_id,
                "species": species_name,
                "species_id": species_id,
            }
            
            # Add raw expression data if requested
            if include_raw:
                gene_names = [g.decode('ascii') for g in h5file['gene_names'][:]]
                spot_data.update({
                    "raw_normed": spot_group['raw_normed'][:],
                    "genes": gene_names,
                    "spatial_coords": spot_group['spatial_coords'][:]
                })
            
            return spot_data
        
        except KeyError as e:
            logger.error(f"Failed to access data for spot {global_idx}: {e}")
            raise ValueError(f"Spot {global_idx} not found in dataset {dataset_info['name']}")
        
        except Exception as e:
            logger.error(f"Error reading data for spot {global_idx}: {e}")
            raise

    def _format_lmdb_key(self, global_idx: int) -> bytes:
        return f"{global_idx:012d}".encode("ascii")

    def _ensure_lmdb_env(self):
        if self._lmdb_env is None:
            with self._lmdb_env_lock:
                if self._lmdb_env is None:
                    env_kwargs = dict(self.lmdb_env_kwargs)
                    self._lmdb_env = lmdb.open(str(self.lmdb_path), **env_kwargs)
        return self._lmdb_env

    def reset_lmdb_env(self):
        if self._lmdb_env is not None:
            try:
                self._lmdb_env.close()
            except Exception:
                pass
        self._lmdb_env = None

    def _get_spot_from_lmdb(self, global_idx: int):
        env = self._ensure_lmdb_env()
        with env.begin(buffers=True) as txn:
            buffer = txn.get(self._format_lmdb_key(global_idx))
        if buffer is None:
            raise BatchSkipException(f"LMDB record missing for idx {global_idx}")
        record = pickle.loads(bytes(buffer))
        meta = record.get("meta", {})
        spot_data = {
            "global_idx": meta.get("global_idx", global_idx),
            "celltype": meta.get("celltype", ""),
            "celltype_id": meta.get("celltype_id", -1),
            "batch_id": meta.get("batch_id", ""),
            "platform": meta.get("platform", "unknown"),
            "platform_id": meta.get("platform_id", 0),
            "organ": meta.get("organ", "unknown"),
            "organ_id": meta.get("organ_id", 0),
            "species": meta.get("species", "unknown"),
            "species_id": meta.get("species_id", 0),
            "gene_ids": record["gene_ids"],
            "raw_normed_values": record["raw_normed_values"],
            "neighbor_indices": record.get("neighbor_indices", []),
        }
        return spot_data

    def prepare_vocabulary(self) -> Any:
        """
        Load gene vocabulary from config file.
        
        The vocabulary file must exist and be valid.
        
        Returns:
            GeneVocab object
        """        
        vocab_path = self.config.vocab_file
        try:
            vocab = GeneVocab.from_file(vocab_path)
            logger.info(f"Loaded vocabulary with {len(vocab)} genes from {vocab_path}")
            return vocab
        except Exception as e:
            logger.error(f"Failed to load gene vocabulary: {e}")
            logger.error(f"Please ensure a valid vocabulary file exists at: {vocab_path}")
            raise RuntimeError(f"Gene vocabulary file not found or invalid: {vocab_path}")


    def close_file_handles(self):
        """Close all open file handles"""
        if hasattr(self, '_dataset_open_files'):
            for dataset_idx, h5file in self._dataset_open_files.items():
                try:
                    h5file.close()
                except Exception as e:
                    logger.warning(f"Failed to close file handle for dataset {dataset_idx}: {e}")
            
            self._dataset_open_files = {}

    def cleanup_resources(self):
        """Clean up all resources"""
        # Close dataset file handles
        if hasattr(self, '_dataset_open_files'):
            for dataset_idx, file_handle in self._dataset_open_files.items():
                try:
                    file_handle.close()
                except Exception as e:
                    logger.warning(f"Failed to close file handle for dataset {dataset_idx}: {e}")
            
            self._dataset_open_files = {}
        
        # Close neighbor file handles
        self.close_neighbors_file_handles()
        
        # Clear caches
        for attr in ['_spot_cache', '_neighbors_cache', '_cache_order']:
            if hasattr(self, attr):
                setattr(self, attr, {})
        
        # Force memory collection
        import gc
        gc.collect()
        
        logger.info("Resources cleaned up")

    def close_neighbors_file_handles(self):
        """Close all neighbor file handles"""
        if hasattr(self, '_neighbors_files_dict'):
            for dataset_idx, file_handle in self._neighbors_files_dict.items():
                try:
                    file_handle.close()
                    logger.debug(f"Closed neighbor file handle for dataset {dataset_idx}")
                except Exception as e:
                    logger.warning(f"Failed to close neighbor file handle for dataset {dataset_idx}: {e}")
            
            self._neighbors_files_dict = {}

    # =========================
    # Runtime overrides (for iterative inference / perturbation propagation)
    # =========================
    def set_runtime_spot_overrides(self, overrides: Dict[int, Dict[str, Any]]) -> None:
        """
        overrides: {global_idx: {"gene_ids": np.ndarray[int], "raw_normed_values": np.ndarray[float]}}
        These values will be used by MemoryEfficientSpotDataset instead of on-disk spot data.
        """
        self._runtime_spot_overrides = overrides or {}

    def clear_runtime_spot_overrides(self) -> None:
        self._runtime_spot_overrides = {}

    def get_runtime_spot_override(self, global_idx: int) -> Optional[Dict[str, Any]]:
        try:
            d = getattr(self, "_runtime_spot_overrides", None) or {}
            return d.get(int(global_idx))
        except Exception:
            return None

    

    
    def get_data_loader(
        self, 
        split: str = "train", 
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 4,
        stratify_by_batch: bool = True,
        validation_split: float = 0.1,
        use_distributed: bool = False,      # Default False, let Trainer manage
        world_size: int = 1,
        rank: int = 0,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,   # Set False for more stability
        is_training: Optional[bool] = None,
    ) -> DataLoader:
        """Get optimized data loader"""
        # Data splitting
        # min_neighbors = getattr(self.config, "min_neighbors", 1)
        # self.build_valid_center_index(min_neighbors=min_neighbors)

        train_indices, val_indices = self.get_data_split(
            validation_split=validation_split,
            stratify_by_batch=stratify_by_batch,
            split=split
        )
        indices = train_indices if split == "train" else val_indices
        #indices = self._filter_invalid_indices(indices, split)
        if len(indices) == 0:
            raise ValueError(f"No valid indices available for split='{split}'. Check LMDB manifest or preprocessing.")

        # Determine training/validation mode
        if is_training is None:
            is_training = (split == "train")

        # Build dataset
        self.dataset = MemoryEfficientSpotDataset(
            indices=indices,
            databank=self,
            max_seq_len=self.config.max_seq_len,
            mask_ratio=self.config.mask_ratio,
            mask_value=self.config.mask_value,
            pad_value=self.config.pad_value,
            pad_token=self.config.pad_token,
            cls_token=self.config.cls_token,
            max_prefetch=batch_size * 10,
            is_training=is_training
        )

        # DataLoader parameters
        loader_kwargs = {
            "batch_size": batch_size,
            "drop_last": drop_last,
            "num_workers": num_workers,
            "collate_fn": optimized_collate_fn,
            "pin_memory": getattr(self.config, "pin_memory", True)
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["worker_init_fn"] = self.build_worker_init_fn()

        sampler = None
        if use_distributed:
            effective_world_size = world_size if world_size and world_size > 0 else int(os.environ.get("WORLD_SIZE", "1"))
            effective_rank = rank if rank is not None else int(os.environ.get("RANK", "0"))
            if effective_world_size <= 0:
                effective_world_size = 1
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=effective_world_size,
                rank=effective_rank,
                shuffle=shuffle if is_training else False,
                drop_last=drop_last
            )
            loader_kwargs["sampler"] = sampler
            loader_kwargs["shuffle"] = False
        else:
            loader_kwargs["shuffle"] = shuffle if is_training else False

        loader = DataLoader(dataset=self.dataset, **loader_kwargs)
        loader.dataset_type = "MemoryEfficientSpotDataset"
        loader = RetryOnSkipDataLoader(loader, max_retries=50)
        logger.info(
            f"Created {split} loader: samples={len(indices)}, batch={batch_size}, "
            f"workers={num_workers}, cache_mode={self.cache_mode}, "
            f"distributed={'yes' if sampler is not None else 'no'}"
        )
        return loader



    def monitor_memory_usage(self):
        """Monitor memory usage statistics."""
        import psutil
        import gc
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
        }
        
        # Add GPU memory info if CUDA is available
        if torch.cuda.is_available():
            stats.update({
                "cuda_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "cuda_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            })
        
        # Number of open file handles
        if hasattr(self, '_dataset_open_files'):
            stats["open_dataset_files"] = len(self._dataset_open_files)
        
        if hasattr(self, '_neighbor_files'):
            stats["open_neighbor_files"] = len(self._neighbor_files)
        
        # Force garbage collection
        gc.collect()
        
        return stats

    def build_worker_init_fn(self):
        """Build DataLoader worker init function to reset cache and random seed."""
        base_seed = int(getattr(self.config, "random_seed", 42))
        databank = self

        def _init_worker(worker_id: int):
            import random as _random
            seed = base_seed + worker_id
            torch.manual_seed(seed)
            np.random.seed(seed)
            _random.seed(seed)
            worker_info = torch.utils.data.get_worker_info()
            dataset = getattr(worker_info, "dataset", None)
            if hasattr(dataset, "reset_worker_state"):
                dataset.reset_worker_state()
            if databank.cache_mode == "lmdb":
                databank.reset_lmdb_env()

        return _init_worker


class MemoryEfficientSpotDataset(Dataset):
    """Memory-optimized spatial transcriptomics dataset"""
    
    def __init__(
        self, 
        indices: List[int],
        databank: 'SpatialDataBank',
        max_seq_len: int = 1024,
        mask_ratio: float = 0.15,
        mask_value: int = -1,
        pad_value: int = -2,
        pad_token: str = "<pad>",
        cls_token: str = "<cls>",
        # append_cls: bool = True,
        max_prefetch: int = 1000,  # Prefetch data size limit
        is_training: bool = True   # Controls whether to apply masking
    ):
        """
        Initialize dataset
        
        Args:
            indices: List of spot indices in current split
            databank: SpatialDataBank instance
            max_seq_len: Maximum sequence length
            mask_ratio: Masking ratio
            mask_value: Mask value
            pad_value: Padding value
            pad_token: Padding token
            cls_token: CLS token
            append_cls: Whether to append CLS token
            max_prefetch: Maximum prefetch data amount
            is_training: Whether in training mode (for masking control)
        """
        self.indices = indices
        self.databank = databank
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.is_training = is_training
        self.include_zero_gene = databank.config.include_zero_gene
        
        # Load vocabulary
        self.vocab = databank.prepare_vocabulary()
        
        # LRU caches
        self.max_prefetch = max_prefetch
        self._spot_cache = {}
        self._neighbors_cache = {}
        self._cache_order = []  # Track access order
        
        # Dataset attributes
        self.num_samples = len(indices)
        
        # Statistics tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.reset_worker_state()
    
    def _update_cache(self, key, value, cache_dict):
        """Update LRU cache"""
        if len(cache_dict) >= self.max_prefetch and key not in cache_dict:
            # Cache full, remove least recently used item
            if self._cache_order:
                oldest = self._cache_order.pop(0)
                if oldest in cache_dict:
                    del cache_dict[oldest]
        
        # Add/update cache
        cache_dict[key] = value
        
        # Update access order
        if key in self._cache_order:
            self._cache_order.remove(key)
        self._cache_order.append(key)
    
    def _get_spot_data(self, global_idx):
        """Get spot data with caching, but always check runtime overrides first"""
        # ALWAYS check runtime overrides first (for iterative inference)
        # This ensures updated overrides are used even if spot was previously cached
        ov = self.databank.get_runtime_spot_override(global_idx) if hasattr(self.databank, "get_runtime_spot_override") else None
        
        if isinstance(ov, dict) and ov.get("gene_ids") is not None and ov.get("raw_normed_values") is not None:
            # Override exists with expression data - get base data for metadata, then override
            # First try cache for metadata
            if global_idx in self._spot_cache:
                base_data = self._spot_cache[global_idx].copy()
            else:
                # Need to fetch base data for metadata
                try:
                    base_data = self.databank.get_spot_data(global_idx)
                    self._update_cache(global_idx, base_data, self._spot_cache)
                except Exception:
                    base_data = {}
            
            # Override expression data while keeping other metadata
            result = {
                "global_idx": global_idx,
                "gene_ids": ov["gene_ids"],
                "raw_normed_values": ov["raw_normed_values"],
                "celltype": base_data.get("celltype", "unknown"),
                "celltype_id": base_data.get("celltype_id", -1),
                "batch_id": base_data.get("batch_id", ""),
                "platform": base_data.get("platform", "unknown"),
                "platform_id": base_data.get("platform_id", 0),
                "organ": base_data.get("organ", "unknown"),
                "organ_id": base_data.get("organ_id", 0),
                "species": base_data.get("species", "unknown"),
                "species_id": base_data.get("species_id", 0),
                "neighbor_indices": base_data.get("neighbor_indices", []),
            }
            return result
        
        # No override, use cache if available
        if global_idx in self._spot_cache:
            self.cache_hits += 1
            return self._spot_cache[global_idx]
        
        self.cache_misses += 1
        data = self.databank.get_spot_data(global_idx)
        self._update_cache(global_idx, data, self._spot_cache)
        return data
    
    def reset_worker_state(self):
        self._spot_cache = {}
        self._neighbors_cache = {}
        self._cache_order = []
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_neighbors(self, global_idx, precomputed=None):
        """Get neighbor data with caching"""
        if precomputed is not None and len(precomputed) > 0:
            return precomputed
        if global_idx in self._neighbors_cache:
            self.cache_hits += 1
            return self._neighbors_cache[global_idx]
        
        self.cache_misses += 1
        neighbors = self.databank.get_neighbors_for_spot(global_idx)
        self._update_cache(global_idx, neighbors, self._neighbors_cache)
        return neighbors
    
    def __len__(self):
        return self.num_samples
    
  

    def __getitem__(self, idx):
        """Get single sample and its neighbors"""
        global_idx = self.indices[idx]
        if self.databank.is_global_idx_invalid(global_idx):
            raise BatchSkipException(f"global_idx {global_idx} marked invalid earlier")

        # Read center spot data
        try:
            center_data = self._get_spot_data(global_idx)
        except Exception as e:
            # Skip batch on any read error to avoid training interruption
            self.databank.mark_invalid_global_idx(global_idx)
            raise BatchSkipException(f"failed to read center spot {global_idx}: {e}")

        # Check for zero expression center (using raw_normed_values)
        # Skip this check in inference mode as masked spots may be initialized to zero
        EPS = 0
        try:
            cvals = center_data["raw_normed_values"]
            import numpy as np
            if self.is_training and np.all(np.abs(np.asarray(cvals)) <= EPS):
                raise BatchSkipException(f"center all-zero (near-zero) raw_normed: {global_idx}")
        except KeyError:
            # Skip if raw_normed_values is missing
            raise BatchSkipException(f"center lacks raw_normed_values: {global_idx}")

        # Get neighbor list and filter out zero-expression neighbors for robustness
        neighbor_indices = list(self._get_neighbors(global_idx, center_data.get("neighbor_indices")))

        neighbor_data = []
        neighbor_global_ids = []
        for n_idx in neighbor_indices:
            if self.databank.is_global_idx_invalid(n_idx):
                continue
            try:
                nd = self._get_spot_data(n_idx)
            except Exception:
                # Skip neighbors that fail to load
                self.databank.mark_invalid_global_idx(n_idx)
                continue

            nvals = nd.get("raw_normed_values", None)
            if nvals is None:
                continue
            
            # Keep zero-expression neighbors in inference mode (may be masked spots)
            # Only filter out zero neighbors in training mode
            if self.is_training and np.all(np.abs(np.asarray(nvals)) <= EPS):
                # Filter out near-zero neighbors (training mode only)
                continue

            neighbor_data.append(nd)
            neighbor_global_ids.append(nd.get("global_idx", n_idx))

        required_neighbors = int(getattr(self.databank.config, "max_neighbors", 0) or 0)
        if len(neighbor_data) == 0 or (required_neighbors > 0 and len(neighbor_data) < required_neighbors):
            self.databank.mark_invalid_global_idx(global_idx)
            raise BatchSkipException(f"center {global_idx} lacks enough neighbors (have {len(neighbor_data)}, need {required_neighbors or '>=1'})")
        if required_neighbors > 0 and len(neighbor_data) > required_neighbors:
            neighbor_data = neighbor_data[:required_neighbors]
            neighbor_global_ids = neighbor_global_ids[:required_neighbors]
            neighbor_indices = neighbor_indices[:required_neighbors]
        else:
            neighbor_indices = neighbor_indices[:len(neighbor_data)]

        # Prepare data for batching
        center_genes = center_data["gene_ids"]
        center_raw_normed = center_data["raw_normed_values"]

        neighbor_genes_list = [n["gene_ids"] for n in neighbor_data]
        neighbor_raw_normed_list = [n["raw_normed_values"] for n in neighbor_data]

        metadata = {
            "center": {
                "global_idx": global_idx,
                "celltype_id": center_data["celltype_id"],
                "batch_id": center_data["batch_id"],
                "platform": center_data.get("platform", "unknown"),
                "platform_id": center_data.get("platform_id", 0),
                "organ": center_data.get("organ", "unknown"),
                "organ_id": center_data.get("organ_id", 0),
                "species": center_data.get("species", "unknown"),
                "species_id": center_data.get("species_id", 0),
            },
            "neighbors": {
                "global_indices": neighbor_global_ids if len(neighbor_global_ids) > 0 else neighbor_indices,
                "count": len(neighbor_data)
            },
            "config": {
                "max_seq_len": self.max_seq_len,
                "pad_token": self.pad_token,
                "pad_value": self.pad_value,
                "cls_token": self.cls_token,
                "mask_ratio": self.mask_ratio,
                "mask_value": self.mask_value,
                "vocab": self.vocab,
                "is_training": self.is_training,
                "include_zero_gene": self.include_zero_gene
            }
        }

        return {
            "center_genes": center_genes,
            "center_raw_normed": center_raw_normed,

            "neighbor_genes": neighbor_genes_list,
            "neighbor_raw_normed": neighbor_raw_normed_list,

            "metadata": metadata
        }
        

class RetryOnSkipDataLoader:
    """Wrapper that retries next batch when collate_fn raises BatchSkipException."""
    def __init__(self, dataloader, max_retries=50):
        self.dataloader = dataloader
        self.max_retries = max_retries

    # Iterator protocol
    def __iter__(self):
        it = iter(self.dataloader)
        while True:
            retries = 0
            while True:
                try:
                    batch = next(it)
                    yield batch
                    break
                except BatchSkipException:
                    retries += 1
                    if retries >= self.max_retries:
                        raise RuntimeError(
                            f"Exceeded max_retries={self.max_retries} due to consecutive invalid batches."
                        )
                    continue
                except StopIteration:
                    return

    def __len__(self):
        return len(self.dataloader)

    # Key proxy properties/methods for Trainer access
    @property
    def dataset(self):
        return self.dataloader.dataset

    @property
    def sampler(self):
        return getattr(self.dataloader, "sampler", None)

    @property
    def batch_size(self):
        return getattr(self.dataloader, "batch_size", None)

    def __getattr__(self, name):
        # Fallback proxy for other attributes (e.g., drop_last, pin_memory)
        return getattr(self.dataloader, name)




def empty_batch_template():
    device = torch.device("cpu")
    L = 2
    flat = {
        "genes": torch.zeros((1, L), dtype=torch.long, device=device),
        "values": torch.zeros((1, L), dtype=torch.float32, device=device),
        "target_values": torch.zeros((1, L), dtype=torch.float32, device=device),
        "raw_normed_values": torch.zeros((1, L), dtype=torch.float32, device=device),
        "raw_normed_target": torch.zeros((1, L), dtype=torch.float32, device=device),
        "attention_mask": torch.ones((1, L), dtype=torch.bool, device=device),
        "padding_attention_mask": torch.ones((1, L), dtype=torch.bool, device=device),

        "nonzero_genes": torch.zeros((1, 1), dtype=torch.long, device=device),
        "nonzero_expr": torch.zeros((1, 1), dtype=torch.float32, device=device),
        "nonzero_raw_normed": torch.zeros((1, 1), dtype=torch.float32, device=device),
        "nonzero_attention_mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
    }
    structure = {
        "center": [{"global_idx": -1, "celltype_id": -1, "batch_id": "empty"}],
        "neighbors": [{"global_indices": [], "count": 0}],
        "batch_to_spots_map": [(0, 0)],
        "center_indices": torch.tensor([0], dtype=torch.long, device=device),
        "platform_ids": torch.zeros((1,), dtype=torch.long, device=device),
        "organ_ids": torch.zeros((1,), dtype=torch.long, device=device),
        "disease_ids": torch.zeros((1,), dtype=torch.long, device=device),
        "species_ids": torch.zeros((1,), dtype=torch.long, device=device),
    }
    return {"flat": flat, "structure": structure, "skip_batch": True}







def pad_sequences_dual(gene_ids_list, raw_normed_values_list, max_len, pad_token_id, pad_value, include_zero_gene=True):
    """
    Enhanced sequence padding function that outputs both full and non-zero expression data.
    
    Args:
        gene_ids_list: List of gene ID arrays
        raw_normed_values_list: List of raw normalized value arrays
        max_len: Maximum sequence length
        pad_token_id: Padding token ID
        pad_value: Padding expression value
        include_zero_gene: Whether to include zero-expression genes (for non-zero data processing)
    
    Returns:
        full_data: Dictionary containing all genes data
        nonzero_data: Dictionary containing only non-zero expression genes data
    """
    batch_size = len(gene_ids_list)
    
    # Check input type
    is_tensor_input = isinstance(gene_ids_list[0], torch.Tensor)
    
    # Process full data (including zeros)
    full_genes = []
    full_raw_normed = []
    
    # Process non-zero expression data
    nonzero_genes = []
    nonzero_raw_normed = []
    
    # Process each sample for both full and non-zero data
    for genes, raw_normed in zip(gene_ids_list, raw_normed_values_list):
        # 1. Save full data directly
        full_genes.append(genes)
        full_raw_normed.append(raw_normed)
        
        # 2. Process non-zero expression data
        if not include_zero_gene:
            if is_tensor_input:
                # For tensor input, use mask filtering
                genes_np = genes.cpu().numpy() if isinstance(genes, torch.Tensor) else genes
                raw_normed_np = raw_normed.cpu().numpy() if isinstance(raw_normed, torch.Tensor) else raw_normed
                
                # Find non-zero expression genes
                non_zero_mask = raw_normed_np > 0
                filtered_genes = genes_np[non_zero_mask]
                filtered_raw_normed = raw_normed_np[non_zero_mask]
                
                # Convert back to tensor
                if len(filtered_genes) > 0:
                    filtered_genes = torch.tensor(filtered_genes, dtype=genes.dtype, device=genes.device)
                    filtered_raw_normed = torch.tensor(filtered_raw_normed, dtype=raw_normed.dtype, device=raw_normed.device)
                else:
                    # If all zeros, keep one gene as placeholder
                    if len(genes) > 0:
                        filtered_genes = genes[0:1]
                        filtered_raw_normed = torch.zeros(1, dtype=raw_normed.dtype, device=raw_normed.device)
                    else:
                        filtered_genes = torch.tensor([0], dtype=genes.dtype, device=genes.device)
                        filtered_raw_normed = torch.zeros(1, dtype=raw_normed.dtype, device=raw_normed.device)
            else:
                # For list input, filter directly
                filtered_data = [(g, r) for g, r in zip(genes, raw_normed) if r > 0]
                if filtered_data:
                    filtered_genes, filtered_raw_normed = zip(*filtered_data)
                else:
                    # If all zeros, keep one gene as placeholder
                    if len(genes) > 0:
                        filtered_genes = [genes[0]]
                        filtered_raw_normed = [0]
                    else:
                        filtered_genes = [0]
                        filtered_raw_normed = [0]
            
            nonzero_genes.append(filtered_genes)
            nonzero_raw_normed.append(filtered_raw_normed)
        else:
            # If including zero genes, use full data directly
            nonzero_genes.append(genes)
            nonzero_raw_normed.append(raw_normed)
    
    # Calculate max sequence length in current batch
    # 1. Full data
    full_max_len = max_len
    
    # 2. Non-zero data
    if not include_zero_gene:
        nonzero_max_len = max([len(genes) for genes in nonzero_genes])
        nonzero_max_len = min(nonzero_max_len, max_len)  # Cap at max_len
    else:
        nonzero_max_len = max_len
    
    # Get correct data types
    if is_tensor_input:
        gene_dtype = gene_ids_list[0].dtype
        raw_normed_dtype = raw_normed_values_list[0].dtype
    else:
        gene_dtype = torch.long
        raw_normed_dtype = torch.float
    
    # Create output tensors
    # 1. Full data tensors
    full_padded_genes = torch.full((batch_size, full_max_len), pad_token_id, dtype=gene_dtype)
    full_padded_raw_normed = torch.full((batch_size, full_max_len), pad_value, dtype=raw_normed_dtype)
    full_attention_mask = torch.zeros((batch_size, full_max_len), dtype=torch.bool)
    
    # 2. Non-zero data tensors
    nonzero_padded_genes = torch.full((batch_size, nonzero_max_len), pad_token_id, dtype=gene_dtype)
    nonzero_padded_raw_normed = torch.full((batch_size, nonzero_max_len), pad_value, dtype=raw_normed_dtype)
    nonzero_attention_mask = torch.zeros((batch_size, nonzero_max_len), dtype=torch.bool)
    
    # Pad each sequence
    # 1. Process full data
    for i, (genes, raw_normed) in enumerate(zip(full_genes, full_raw_normed)):
        seq_genes = genes if is_tensor_input else list(genes)
        seq_raw_normed = raw_normed if is_tensor_input else list(raw_normed)
        
        # Truncate sequences exceeding max length
        if len(seq_genes) > full_max_len:
            seq_genes = seq_genes[:full_max_len]
            seq_raw_normed = seq_raw_normed[:full_max_len]
        
        # Calculate actual length
        L = len(seq_genes)
        
        # Apply padding
        if is_tensor_input:
            full_padded_genes[i, :L] = seq_genes
            full_padded_raw_normed[i, :L] = seq_raw_normed
        else:
            full_padded_genes[i, :L] = torch.tensor(seq_genes, dtype=gene_dtype)
            full_padded_raw_normed[i, :L] = torch.tensor(seq_raw_normed, dtype=raw_normed_dtype)
        
        # Mark non-padding positions
        full_attention_mask[i, :L] = True
    
    # 2. Process non-zero data
    for i, (genes, raw_normed) in enumerate(zip(nonzero_genes, nonzero_raw_normed)):
        seq_genes = genes if is_tensor_input else list(genes)
        seq_raw_normed = raw_normed if is_tensor_input else list(raw_normed)
        
        # Truncate sequences exceeding max length
        if len(seq_genes) > nonzero_max_len:
            seq_genes = seq_genes[:nonzero_max_len]
            seq_raw_normed = seq_raw_normed[:nonzero_max_len]
        
        # Calculate actual length
        L = len(seq_genes)
        
        # Apply padding
        if is_tensor_input:
            nonzero_padded_genes[i, :L] = seq_genes
            nonzero_padded_raw_normed[i, :L] = seq_raw_normed
        else:
            nonzero_padded_genes[i, :L] = torch.tensor(seq_genes, dtype=gene_dtype)
            nonzero_padded_raw_normed[i, :L] = torch.tensor(seq_raw_normed, dtype=raw_normed_dtype)
        
        # Mark non-padding positions
        nonzero_attention_mask[i, :L] = True
    
    # Build return data
    full_data = {
        "gene_ids": full_padded_genes,
        "expr_values": full_padded_raw_normed,
        "attention_mask": full_attention_mask
    }
    
    nonzero_data = {
        "gene_ids": nonzero_padded_genes,
        "expr_values": nonzero_padded_raw_normed,
        "attention_mask": nonzero_attention_mask
    }
    
    return full_data, nonzero_data



def optimized_collate_fn(batch):
    """Memory-optimized batch processing function with platform support."""
    # Get config
    config = batch[0]["metadata"]["config"]
    is_training = config.get("is_training", True)
    
    # In inference mode, allow zero-expression centers (e.g., masked spots initialized to zero)
    # Only skip zero-expression centers in training mode
    if is_training:
        # Filter based on center expression: skip only if ALL centers are zero
        EPS_CENTER = 0
        valid_indices = []
        for i, item in enumerate(batch):
            try:
                if not _is_all_zero(item["center_raw_normed"], eps=EPS_CENTER):
                    valid_indices.append(i)
            except Exception:
                # Safety fallback: treat read failures as invalid centers
                pass
        if not valid_indices:
            # All centers are zero -> skip this batch
            raise BatchSkipException("all centers are zero (or invalid) in this batch")
        if len(valid_indices) < len(batch):
            # Remove zero centers, keep only valid centers for batching
            batch = [batch[i] for i in valid_indices]
    # else: In inference mode, keep all centers including zero-expression masked spots
    
    # Extract center spot data
    center_geneIDs = [item["center_genes"] for item in batch]
    center_raw_normed = [item["center_raw_normed"] for item in batch]
    platform_ids = [item["metadata"]["center"].get("platform_id", 0) for item in batch]
    organ_ids = [item["metadata"]["center"].get("organ_id", 0) for item in batch]
    disease_ids = [item["metadata"]["center"].get("disease_id", 0) for item in batch]
    species_ids = [item["metadata"]["center"].get("species_id", 0) for item in batch]
    
    # Prepare merged data structures
    all_geneIDs = []
    all_raw_normed = []
    all_platform_ids = []
    all_organ_ids = []
    all_disease_ids = []
    all_species_ids = []
    # Record structure information
    batch_size = len(batch)
    center_indices = list(range(batch_size))
    batch_to_spots_map = []
    
    # Collect center and neighbor global indices
    centers_global_indices = [item["metadata"]["center"]["global_idx"] for item in batch]
    neighbors_global_indices_per_center = [
        item["metadata"]["neighbors"]["global_indices"] for item in batch
    ]
    
    # Current position
    current_idx = batch_size
    
    # Initialize batch-local row to global index mapping (centers first)
    batch_local_to_global = list(centers_global_indices)
    neighbors_local_rows_per_center = []

    # First add all center spots
    all_geneIDs.extend(center_geneIDs)
    all_raw_normed.extend(center_raw_normed)
    all_platform_ids.extend(platform_ids)
    all_organ_ids.extend(organ_ids)
    all_disease_ids.extend(disease_ids)
    
    # Then add neighbors
    for batch_idx, item in enumerate(batch):
        neighbor_geneIDs = item["neighbor_genes"]
        neighbor_raw_normed = item["neighbor_raw_normed"]
        platform_id = platform_ids[batch_idx]
        organ_id = organ_ids[batch_idx]
        disease_id = disease_ids[batch_idx]
        species_id = species_ids[batch_idx]
        start_idx = current_idx
        
        # Add all neighbors
        all_geneIDs.extend(neighbor_geneIDs)
        all_raw_normed.extend(neighbor_raw_normed)
        all_platform_ids.extend([platform_id] * len(neighbor_geneIDs))
        all_organ_ids.extend([organ_id] * len(neighbor_geneIDs))
        all_disease_ids.extend([disease_id] * len(neighbor_geneIDs))
        all_species_ids.extend([species_id] * len(neighbor_geneIDs))
        
        # Append neighbor global indices to batch_local_to_global
        neigh_global = neighbors_global_indices_per_center[batch_idx]
        # Defensive trim to actual neighbor count
        take_n = len(neighbor_geneIDs)
        neigh_global = neigh_global[:take_n]
        # Record neighbor batch-local row indices
        neigh_local_rows = list(range(current_idx, current_idx + take_n))
        neighbors_local_rows_per_center.append(neigh_local_rows)
        # Append to mapping table
        batch_local_to_global.extend([int(g) for g in neigh_global])

        # Update current position
        current_idx += len(neighbor_geneIDs)
        end_idx = current_idx
        
        # Record batch spot range
        batch_to_spots_map.append((start_idx, end_idx))
    
    # Get full data and non-zero expression data
    cls_token_id = config["vocab"][config["cls_token"]] if config.get("append_cls", False) else None
    full_data, nonzero_data = pad_sequences_dual(
        all_geneIDs,
        all_raw_normed,
        max_len=config["max_seq_len"],
        pad_token_id=config["vocab"][config["pad_token"]],
        pad_value=config["pad_value"],
        include_zero_gene=True  # Full data includes zero genes
    )
    
    # Get non-zero expression data
    _, nonzero_only_data = pad_sequences_dual(
        all_geneIDs,
        all_raw_normed,
        max_len=config["max_seq_len"],
        pad_token_id=config["vocab"][config["pad_token"]],
        pad_value=config["pad_value"],
        include_zero_gene=False  # Non-zero data only includes non-zero genes
    )
    
    # Filter by center rows only (avoid discarding batch due to some all-zero neighbors)
    # Skip this check in inference mode as masked spots may be initialized to zero
    eff_mask = full_data["attention_mask"].to(torch.bool)         # [num_spots, max_seq_len]
    vals_for_model = full_data["expr_values"].to(torch.float32)
    EPS = 0
    nz_eff = ((vals_for_model.abs() > EPS) & eff_mask).sum(dim=1)
    centers_nz = nz_eff[:batch_size] > 0
    
    if is_training:
        # Training mode: filter zero centers
        if not bool(centers_nz.all()):
            kept_indices = [i for i, v in enumerate(centers_nz.tolist()) if v]
            if len(kept_indices) == 0:
                # All centers in batch are zero: signal to retry next batch
                raise BatchSkipException("all centers are zero after padding; retry next batch")
            # Recursive call with only valid centers
            reduced_batch = [batch[i] for i in kept_indices]
            return optimized_collate_fn(reduced_batch)

        # Skip if any position is all-zero to avoid sending invalid global_idx to model
        if (nz_eff == 0).any().item():
            placeholder = empty_batch_template()
            placeholder["skip_batch"] = True
            return placeholder
    # else: In inference mode, keep all spots including zero-expression masked spots
        
    # Extract structure information
    dtype = full_data["gene_ids"].dtype
    tokenized_data = {
        # Full data
        "genes": full_data["gene_ids"],                        # [num_spots, max_seq_len]
        "values": full_data["expr_values"],                    # [num_spots, max_seq_len]
        "raw_normed_values": full_data["expr_values"],         # [num_spots, max_seq_len]
        "padding_attention_mask": full_data["attention_mask"], # [num_spots, max_seq_len]
        
        # Non-zero expression data
        "nonzero_genes": nonzero_data["gene_ids"],                     # [num_spots, nonzero_max_len]
        "nonzero_expr": nonzero_data["expr_values"],                   # [num_spots, nonzero_max_len]
        "nonzero_raw_normed": nonzero_data["expr_values"],             # [num_spots, nonzero_max_len]
        "nonzero_attention_mask": nonzero_data["attention_mask"]       # [num_spots, nonzero_max_len]
    }
    
    platform_ids = torch.tensor(all_platform_ids,dtype= dtype)
    organ_ids = torch.tensor(all_organ_ids,dtype= dtype)
    species_ids = torch.tensor(all_species_ids,dtype= dtype)

    disease_ids = torch.tensor(all_disease_ids,dtype= dtype)


    structure_data = {
        "center": [item["metadata"]["center"] for item in batch],
        "neighbors": [item["metadata"]["neighbors"] for item in batch],
        "batch_to_spots_map": batch_to_spots_map,
        "center_indices": center_indices,
        "platform_ids": platform_ids,
        "organ_ids": organ_ids,
        "disease_ids": disease_ids,
        "species_ids": species_ids,
        "batch_local_to_global": torch.as_tensor(batch_local_to_global, dtype=torch.long),
        "centers_global_indices": torch.as_tensor(centers_global_indices, dtype=torch.long),
        "neighbors_local_rows_per_center": neighbors_local_rows_per_center
    }
    
    
    return {
        "flat": tokenized_data,
        "structure": structure_data
    }


def _is_all_zero(seq, eps=1e-12):
    """Check if sequence is all zeros. Supports list/np.ndarray/torch.Tensor."""
    if isinstance(seq, torch.Tensor):
        return (seq.abs() <= eps).all().item()
    else:
        import numpy as np
        arr = np.asarray(seq)
        return np.all(np.abs(arr) <= eps)


class BatchSkipException(Exception):
    """Signal to skip current batch and fetch next one"""
    pass


def empty_batch_template():
    """Return a minimal valid empty batch template"""
    return {
        "flat": {
            "genes": torch.zeros((1, 2), dtype=torch.long),
            "values": torch.zeros((1, 2), dtype=torch.float),
            "target_values": torch.zeros((1, 2), dtype=torch.float),
            "raw_normed_values": torch.zeros((1, 2), dtype=torch.float),
            "raw_normed_target": torch.zeros((1, 2), dtype=torch.float),
            "attention_mask": torch.ones((1, 2), dtype=torch.bool),
            "values": torch.zeros((1, 2), dtype=torch.float),
            "padding_attention_mask": torch.ones((1, 2), dtype=torch.bool),
            "nonzero_genes": torch.zeros((1, 1), dtype=torch.long),
            "nonzero_expr": torch.zeros((1, 1), dtype=torch.float),
            "nonzero_raw_normed": torch.zeros((1, 1), dtype=torch.float),
            "nonzero_attention_mask": torch.zeros((1, 1), dtype=torch.bool),
        },
        "structure": {
            "center": [{"global_idx": -1, "celltype_id": -1, "batch_id": "empty"}],
            "neighbors": [{"global_indices": [], "count": 0}],
            "batch_to_spots_map": [(0, 0)],
            "center_indices": [0],
            "platform_ids": torch.zeros((1,), dtype=torch.long),
            "organ_ids": torch.zeros((1,), dtype=torch.long),
            "disease_ids": torch.zeros((1,), dtype=torch.long),
            "species_ids": torch.zeros((1,), dtype=torch.long),
        },
        "skip_batch": True  # Flag to indicate batch should be skipped
    }



