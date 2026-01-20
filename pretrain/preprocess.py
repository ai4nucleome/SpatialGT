"""
SpatialGT Data Preprocessing Script

This script handles preprocessing of spatial transcriptomics data including:
- Basic data preprocessing (normalization, log1p transformation)
- Spatial neighbor computation
- Spot data caching for efficient training

Usage:
    python preprocess.py --dataset_list ./datalist.txt --cache_dir ./preprocessed_cache
"""

import os
import sys
import time
import json
import glob
import logging
import argparse
import datetime
import traceback
from pathlib import Path

import h5py
import torch
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

# Local imports
from utils import setup_logger
from Config import Config
from spatial_databank import SpatialDataBank


def setup_preprocess_logger():
    """Set up logger for preprocessing."""
    logger = logging.getLogger('stGPT_preprocess')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('preprocess.log')
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


logger = setup_preprocess_logger()


def compute_spatial_neighbors(spatial_coords, max_neighbors, use_gpu=True, block_size=5000):
    """
    Compute spatial neighbors using GPU acceleration.
    
    Args:
        spatial_coords: numpy array of shape (n_spots, 2) with spatial coordinates
        max_neighbors: Maximum number of neighbors per spot
        use_gpu: Whether to use GPU acceleration
        block_size: Block size for processing to avoid memory overflow
    
    Returns:
        all_neighbors: dict mapping spot index to neighbor indices
        all_distances: dict mapping spot index to neighbor distances
    """
    n_spots = spatial_coords.shape[0]
    
    # Use only first 2 dimensions for spatial coordinates
    if spatial_coords.shape[1] > 2:
        spatial_coords = spatial_coords[:, :2]
    
    logger.info(f"Computing {max_neighbors} nearest neighbors for {n_spots} spots")
    
    # GPU acceleration setup
    if use_gpu and torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            gpu_info = f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
            logger.info(f"Using GPU acceleration - {gpu_info}")
            
            # Transfer coordinates to GPU
            spatial_coords_gpu = torch.tensor(spatial_coords, dtype=torch.float32, device=device)
            logger.info(f"Coordinates transferred to GPU, shape: {spatial_coords_gpu.shape}")
            
        except Exception as e:
            logger.warning(f"GPU initialization failed: {str(e)}, falling back to CPU")
            use_gpu = False
    else:
        use_gpu = False
        logger.info("Using CPU for distance computation")
    
    # Storage for neighbor information
    all_neighbors = {}
    all_distances = {}
    
    # Process in blocks to avoid memory overflow
    n_blocks = (n_spots + block_size - 1) // block_size
    logger.info(f"Processing in {n_blocks} blocks, {block_size} spots per block")
    
    for i in range(n_blocks):
        start_i = i * block_size
        end_i = min(start_i + block_size, n_spots)
        
        logger.info(f"Processing block {i+1}/{n_blocks} ({start_i}:{end_i})")
        start_time = time.time()
        
        if use_gpu:
            # GPU accelerated version
            for j in range(start_i, end_i):
                point = spatial_coords_gpu[j].view(1, -1)
                diffs = spatial_coords_gpu - point
                sq_dists = torch.sum(diffs * diffs, dim=1)
                sq_dists[j] = float('inf')  # Exclude self
                
                _, nearest_indices = torch.topk(sq_dists, k=max_neighbors, largest=False)
                nearest_indices = nearest_indices.cpu().numpy()
                nearest_dists = torch.sqrt(sq_dists[nearest_indices]).cpu().numpy()
                
                all_neighbors[j] = nearest_indices
                all_distances[j] = nearest_dists
        else:
            # CPU version
            for j in range(start_i, end_i):
                dists = np.sqrt(np.sum((spatial_coords - spatial_coords[j])**2, axis=1))
                dists[j] = np.inf  # Exclude self
                
                nearest_indices = np.argsort(dists)[:max_neighbors]
                nearest_dists = dists[nearest_indices]
                
                all_neighbors[j] = nearest_indices
                all_distances[j] = nearest_dists
        
        elapsed = time.time() - start_time
        logger.info(f"Block {i+1} completed in {elapsed:.2f} seconds")
        
        # Periodically clear GPU cache
        if use_gpu and i % 5 == 4:
            torch.cuda.empty_cache()
    
    # Final GPU cache cleanup
    if use_gpu:
        torch.cuda.empty_cache()
    
    return all_neighbors, all_distances


def create_sparse_adjacency_matrix(all_neighbors, n_spots):
    """
    Create sparse adjacency matrix from neighbor information.
    
    Args:
        all_neighbors: dict mapping spot index to neighbor indices
        n_spots: Total number of spots
    
    Returns:
        Sparse CSR adjacency matrix
    """
    rows = []
    cols = []
    
    for i in range(n_spots):
        if i in all_neighbors:
            for j in all_neighbors[i]:
                rows.append(i)
                cols.append(j)
    
    data = np.ones(len(rows), dtype=float)
    adj_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_spots, n_spots))
    
    return adj_sparse


def save_neighbors_to_hdf5(neighbors_file, dataset_name, dataset_idx, max_neighbors, 
                           start_idx, end_idx, all_neighbors, all_distances):
    """
    Save neighbor relationships to HDF5 file.
    
    Args:
        neighbors_file: Path to output file
        dataset_name: Name of the dataset
        dataset_idx: Index of the dataset
        max_neighbors: Maximum number of neighbors
        start_idx: Starting global index
        end_idx: Ending global index
        all_neighbors: dict of neighbor indices
        all_distances: dict of neighbor distances
    """
    n_spots = end_idx - start_idx
    
    with h5py.File(neighbors_file, 'w') as f:
        # Set file attributes
        f.attrs['dataset_name'] = dataset_name
        f.attrs['dataset_idx'] = dataset_idx
        f.attrs['max_neighbors'] = max_neighbors
        f.attrs['start_idx'] = start_idx
        f.attrs['end_idx'] = end_idx
        f.attrs['n_spots'] = n_spots
        f.attrs['created_at'] = datetime.datetime.now().isoformat()
        f.attrs['version'] = 'v3'
        
        # Create groups for neighbors and distances
        neighbors_group = f.create_group('neighbors')
        distances_group = f.create_group('distances')
        
        # Statistics
        neighbor_counts = []
        total_distances = []
        
        # Save each spot's neighbors and distances
        for local_idx in range(n_spots):
            global_idx = local_idx + start_idx
            
            if local_idx in all_neighbors:
                spot_neighbors = all_neighbors[local_idx].tolist()
                spot_distances = all_distances[local_idx].tolist()
            else:
                spot_neighbors = []
                spot_distances = []
            
            neighbor_counts.append(len(spot_neighbors))
            total_distances.extend(spot_distances)
            
            # Pad to fixed length
            padded_neighbors = spot_neighbors + [-1] * (max_neighbors - len(spot_neighbors))
            padded_distances = spot_distances + [0.0] * (max_neighbors - len(spot_distances))
            
            # Convert to global indices
            global_neighbors = [n + start_idx if n >= 0 else -1 for n in padded_neighbors]
            
            # Save neighbors
            neighbors_group.create_dataset(
                str(global_idx),
                data=np.array(global_neighbors, dtype=np.int32),
                compression="gzip",
                compression_opts=4
            )
            
            # Save distances
            distances_group.create_dataset(
                str(global_idx),
                data=np.array(padded_distances, dtype=np.float32),
                compression="gzip",
                compression_opts=4
            )
        
        # Save statistics
        stats_group = f.create_group('statistics')
        stats_group.create_dataset('neighbor_counts', data=np.array(neighbor_counts, dtype=np.int32))
        
        if total_distances:
            stats_group.create_dataset('all_distances', data=np.array(total_distances, dtype=np.float32))
            
            # Compute distance statistics
            avg_distance = float(np.mean(total_distances))
            std_distance = float(np.std(total_distances))
            min_distance = float(np.min(total_distances))
            max_distance = float(np.max(total_distances))
            
            f.attrs['avg_distance'] = avg_distance
            f.attrs['std_distance'] = std_distance
            f.attrs['min_distance'] = min_distance
            f.attrs['max_distance'] = max_distance
        
        # Neighbor count statistics
        if neighbor_counts:
            avg_neighbors = float(np.mean(neighbor_counts))
            max_neighbors_found = int(np.max(neighbor_counts))
        else:
            avg_neighbors = 0.0
            max_neighbors_found = 0
            
        f.attrs['avg_neighbors'] = avg_neighbors
        f.attrs['max_neighbors_found'] = max_neighbors_found
    
    logger.info(f"Neighbor relationships saved to: {neighbors_file}")
    if neighbor_counts:
        logger.info(f"Average neighbors per spot: {avg_neighbors:.2f}")
    if total_distances:
        logger.info(f"Average distance: {avg_distance:.4f}, std: {std_distance:.4f}")


def process_dataset_neighbors(databank, dataset_idx, cache_dir, config, force_rebuild=False, use_gpu=True):
    """
    Process neighbor relationships for a single dataset.
    
    Args:
        databank: SpatialDataBank instance
        dataset_idx: Dataset index
        cache_dir: Cache directory path
        config: Configuration object
        force_rebuild: Whether to force rebuild
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Path to the neighbors file
    """
    dataset_info = databank.metadata["datasets"][dataset_idx]
    dataset_name = dataset_info["name"]
    
    # Set neighbor file path
    neighbors_dir = Path(cache_dir) / "neighbors"
    neighbors_dir.mkdir(exist_ok=True)
    dataset_neighbors_file = neighbors_dir / f"neighbors_{dataset_name}_n{config.max_neighbors}_v3.h5"
    
    # Check if file already exists
    if dataset_neighbors_file.exists() and not force_rebuild:
        logger.info(f"Neighbors file already exists for dataset {dataset_name}: {dataset_neighbors_file}")
        return dataset_neighbors_file
    
    logger.info(f"Computing neighbors for dataset {dataset_name}")
    
    try:
        # Get dataset
        adata, _ = databank._get_dataset_by_idx(dataset_idx)
        
        # Get dataset range
        start_idx = databank.metadata["dataset_indices"][dataset_idx]["start_idx"]
        end_idx = databank.metadata["dataset_indices"][dataset_idx]["end_idx"]
        n_spots = end_idx - start_idx
        
        # Check spatial coordinates
        if 'spatial' not in adata.obsm or adata.obsm['spatial'] is None:
            logger.error(f"Dataset {dataset_name} has no valid spatial coordinates")
            return None
        
        spatial_coords = adata.obsm['spatial']
        logger.info(f"Dataset {dataset_name}: {n_spots} spots, spatial shape: {spatial_coords.shape}")
        
        # Compute neighbors
        all_neighbors, all_distances = compute_spatial_neighbors(
            spatial_coords, 
            config.max_neighbors, 
            use_gpu, 
            block_size=5000
        )
        
        # Create sparse adjacency matrix
        adj_sparse = create_sparse_adjacency_matrix(all_neighbors, n_spots)
        
        # Save sparse adjacency matrix to dataset cache directory
        adj_dir = Path(dataset_info["cache_dir"])
        adj_dir.mkdir(exist_ok=True, parents=True)
        adj_file_sparse = adj_dir / "adjacency_sparse.npz"
        adj_file = adj_dir / "adjacency.npz"
        
        sp.save_npz(adj_file_sparse, adj_sparse)
        
        # Save marker file for compatibility
        with open(adj_file, 'wb') as f:
            np.savez_compressed(f, is_sparse=True, shape=adj_sparse.shape)
        
        logger.info(f"Sparse adjacency matrix saved: {adj_file_sparse}")

        # Save detailed neighbor info to HDF5
        save_neighbors_to_hdf5(
            dataset_neighbors_file, dataset_name, dataset_idx, 
            config.max_neighbors, start_idx, end_idx, 
            all_neighbors, all_distances
        )
        
        return dataset_neighbors_file
        
    except Exception as e:
        logger.error(f"Error processing neighbors for dataset {dataset_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def rebuild_neighbor_index(databank, cache_dir, config):
    """
    Rebuild neighbor index file without recomputing neighbors.
    
    Scans the cache directory for existing neighbor files and generates a new index.
    
    Args:
        databank: SpatialDataBank instance
        cache_dir: Cache directory path
        config: Configuration object
    """
    logger.info("Rebuilding neighbor index file...")
    neighbor_files = {}
    neighbors_dir = Path(cache_dir) / "neighbors"

    if not databank.metadata["datasets"]:
        logger.error("No datasets found in databank, cannot rebuild index.")
        return

    for dataset_idx, dataset_info in enumerate(databank.metadata["datasets"]):
        dataset_name = dataset_info["name"]
        expected_neighbor_file = neighbors_dir / f"neighbors_{dataset_name}_n{config.max_neighbors}_v3.h5"

        if expected_neighbor_file.exists():
            neighbor_files[str(dataset_idx)] = str(expected_neighbor_file)
            logger.info(f"Found neighbors file for '{dataset_name}': {expected_neighbor_file}")
        else:
            logger.warning(f"Neighbors file not found for '{dataset_name}': {expected_neighbor_file}")

    if not neighbor_files:
        logger.error(f"No matching v3 neighbor files found in {neighbors_dir}, cannot create index.")
        return

    # Create index file
    index_file = neighbors_dir / "neighbors_index.json"
    try:
        with open(index_file, 'w') as f:
            json.dump(neighbor_files, f, indent=2)
        logger.info(f"Neighbor index file successfully rebuilt: {index_file}")
        logger.info(f"Indexed {len(neighbor_files)} datasets.")
    except IOError as e:
        logger.error(f"Error writing index file {index_file}: {e}")


def main_preprocess():
    """Complete data preprocessing pipeline."""
    # Argument parsing
    parser = argparse.ArgumentParser(description="SpatialGT Data Preprocessing Tool")
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--cache_dir', type=str, default='./preprocessed_cache', help='Cache directory')
    parser.add_argument('--force_rebuild', action='store_true', help='Force rebuild all cache')
    parser.add_argument('--rebuild_index', action='store_true', help='Only rebuild neighbor index file')
    parser.add_argument('--dataset_list', type=str, default="./datalist.txt", help='Dataset list file')
    parser.add_argument('--memory_limit', type=int, default=32, help='Memory limit (GB)')
    parser.add_argument('--max_neighbors', type=int, default=8, help='Maximum number of neighbors')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # GPU usage setting
    use_gpu = not args.no_gpu
    if not use_gpu:
        logger.info("GPU acceleration disabled")
    
    # Load configuration
    config = Config()
    if args.config:
        config = Config.load(args.config)
    
    # Override config with command line arguments
    config.max_neighbors = args.max_neighbors
    
    # Get subdirectories helper
    def get_subdirectories(path):
        """Get all subdirectory paths under the specified path."""
        path = Path(path)
        return [str(p) for p in path.iterdir() if p.is_dir()]
    
    # Load dataset paths from dataset_list
    dataset_paths = []
    if args.dataset_list and os.path.exists(args.dataset_list):
        with open(args.dataset_list, 'r') as f:
            folder_paths = [line.strip() for line in f if line.strip()]
        
        # Find all .h5ad files in each folder
        for folder in folder_paths:
            h5ad_files = glob.glob(os.path.join(folder, "*.h5ad"))
            if h5ad_files:
                dataset_paths.extend(h5ad_files)
            else:
                logger.warning(f"No .h5ad files found in: {folder}")
        
        logger.info(f"Found {len(dataset_paths)} preprocessed h5ad files")
    else:
        logger.warning("Dataset list not specified or file not found, using default path")
        raise ValueError("Please provide a valid dataset_list file")
    
    # Validate datasets
    if not dataset_paths:
        logger.error("No valid h5ad files found, please check data path settings")
        logger.error(f"dataset_list parameter: {args.dataset_list}")
        return

    # Initialize databank
    logger.info(f"Initializing SpatialDataBank with {len(dataset_paths)} datasets")
    databank = SpatialDataBank(
        dataset_paths=dataset_paths,
        cache_dir=args.cache_dir,
        config=config,
        force_rebuild=args.force_rebuild
    )
    
    # If rebuild_index is specified, only rebuild index and exit
    if args.rebuild_index:
        rebuild_neighbor_index(databank, args.cache_dir, config)
        logger.info("Neighbor index rebuild complete.")
        return

    # ===== Phase 1: Basic Data Preprocessing =====
    logger.info("=== Phase 1: Basic Data Preprocessing ===")
    
    for dataset_idx, dataset_info in enumerate(databank.metadata["datasets"]):
        dataset_name = dataset_info["name"]
        logger.info(f"Processing dataset {dataset_idx+1}/{len(databank.metadata['datasets'])}: {dataset_name}")
        
        try:
            adata, _ = databank._get_dataset_by_idx(dataset_idx)
            logger.info(f"Dataset {dataset_name} basic preprocessing complete")
        except Exception as e:
            logger.error(f"Dataset {dataset_name} preprocessing failed: {str(e)}")
            continue
    
    # ===== Phase 2: Spatial Neighbor Computation =====
    logger.info("=== Phase 2: Computing Spatial Neighbors ===")
    
    neighbor_files = {}
    
    for dataset_idx, dataset_info in enumerate(databank.metadata["datasets"]):
        dataset_name = dataset_info["name"]
        logger.info(f"Processing neighbors for dataset {dataset_name}")
        
        neighbors_file = process_dataset_neighbors(
            databank, dataset_idx, args.cache_dir, config, 
            force_rebuild=args.force_rebuild, use_gpu=use_gpu
        )
        
        if neighbors_file:
            neighbor_files[str(dataset_idx)] = str(neighbors_file)
            logger.info(f"Dataset {dataset_name} neighbor processing complete")
        else:
            logger.error(f"Dataset {dataset_name} neighbor processing failed")
    
    # Create neighbor index file
    neighbors_dir = Path(args.cache_dir) / "neighbors"
    index_file = neighbors_dir / "neighbors_index.json"

    neighbors_dir.mkdir(parents=True, exist_ok=True)
    
    if len(neighbor_files) == 0:
        logger.error(
            "No neighbor files generated (possibly all datasets became invalid after preprocessing). "
            "Created empty neighbors/ directory and skipping subsequent phases."
        )
        with open(index_file, 'w') as f:
            json.dump({}, f, indent=2)
        return
    
    with open(index_file, 'w') as f:
        json.dump(neighbor_files, f, indent=2)
    
    logger.info(f"Neighbor index file saved: {index_file}")
    
    # ===== Phase 3: Precompute Spot Data =====
    logger.info("=== Phase 3: Precomputing Spot Data ===")
    
    for dataset_idx, dataset_info in enumerate(databank.metadata["datasets"]):
        dataset_name = dataset_info["name"]
        logger.info(f"Preprocessing spot data for dataset {dataset_name}")
        
        try:
            # Create spot data file
            spots_file = databank.preprocess_dataset_spots(dataset_idx)
            logger.info(f"Dataset {dataset_name} spot data preprocessing complete: {spots_file}")
            
            # Cleanup resources
            databank.cleanup_resources()
            
            # Monitor memory usage
            mem_stats = databank.monitor_memory_usage()
            logger.info(f"Memory usage: RSS {mem_stats['rss_mb']:.2f}MB, VMS {mem_stats['vms_mb']:.2f}MB")
            
            # Memory limit check
            if mem_stats['rss_mb'] > args.memory_limit * 1024:
                logger.warning(f"Memory usage exceeds limit ({args.memory_limit}GB), performing cleanup")
                databank.cleanup_resources()
                
        except Exception as e:
            logger.error(f"Dataset {dataset_name} spot data preprocessing failed: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    # ===== Phase 4: Generate Final Metadata =====
    logger.info("=== Phase 4: Generating Final Metadata ===")
    
    # Save complete processing statistics
    processing_stats = {
        'version': 'v3',
        'processing_time': datetime.datetime.now().isoformat(),
        'total_datasets': len(databank.metadata["datasets"]),
        'successful_neighbors': len(neighbor_files),
        'config': {
            'max_neighbors': config.max_neighbors,
            'use_gpu': use_gpu,
            'memory_limit': args.memory_limit
        },
        'cache_dir': args.cache_dir
    }
    
    # Final cleanup
    databank.cleanup_resources()
    
    logger.info("=== Data Preprocessing Successfully Complete! ===")
    logger.info(f"Processed {len(databank.metadata['datasets'])} datasets")
    logger.info(f"Successfully computed neighbors for {len(neighbor_files)} datasets")
    logger.info(f"Cache directory: {args.cache_dir}")


if __name__ == "__main__":
    try:
        main_preprocess()
    except Exception as e:
        logger.exception(f"Error during preprocessing: {e}")
        sys.exit(1)
