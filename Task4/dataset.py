"""
Storm Lightning Prediction Dataset Module

Provides PyTorch Dataset and DataLoader utilities for training lightning prediction models.
Supports both coordinate-based and density-map based targets.

Usage:
    # Coordinate-based targets (t, x, y per strike)
    from dataset import StormDataset, create_datasets, create_dataloaders
    train_ds, val_ds, test_ds = create_datasets(subset_size=200)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=4, max_strikes=50000
    )

    # Density-map targets
    from dataset import StormDensityDataset, create_density_datasets, create_density_dataloaders
    train_ds, val_ds, test_ds = create_density_datasets(subset_size=200, sigma=3.0)
    train_loader, val_loader, test_loader = create_density_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=4
    )
"""

import pandas as pd
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import random


# =============================================================================
# Base Dataset Class
# =============================================================================

class BaseStormDataset(Dataset):
    """Base class for storm datasets with common image processing."""

    def _process_images(self, event: dict) -> np.ndarray:
        """Resize and normalize image modalities to (4, 36, 192, 192)."""
        vis_resized = np.zeros((192, 192, 36), dtype=np.float32)
        vil_resized = np.zeros((192, 192, 36), dtype=np.float32)

        for i in range(36):
            vis_resized[:, :, i] = np.array(
                Image.fromarray(event['vis'][:, :, i]).resize((192, 192), Image.BILINEAR)
            )
            vil_resized[:, :, i] = np.array(
                Image.fromarray(event['vil'][:, :, i]).resize((192, 192), Image.BILINEAR)
            )

        ir069_data = event['ir069'].astype(np.float32)
        ir107_data = event['ir107'].astype(np.float32)

        vis_permuted = np.transpose(vis_resized, (2, 0, 1))
        ir069_permuted = np.transpose(ir069_data, (2, 0, 1))
        ir107_permuted = np.transpose(ir107_data, (2, 0, 1))
        vil_permuted = np.transpose(vil_resized, (2, 0, 1))

        vis_norm = vis_permuted / 10000.0
        ir069_norm = (ir069_permuted + 8000.0) / 7000.0
        ir107_norm = (ir107_permuted + 7000.0) / 9000.0
        vil_norm = vil_permuted / 255.0

        images = np.stack([vis_norm, ir069_norm, ir107_norm, vil_norm], axis=0).astype(np.float32)

        return images


# =============================================================================
# Coordinate-based Dataset
# =============================================================================

class StormDataset(BaseStormDataset):
    """
    PyTorch Dataset for storm prediction with coordinate targets.

    Loads and preprocesses storm events on-the-fly from HDF5 file.
    Resizes vis and vil (384x384) to match ir069 and ir107 (192x192).
    Normalizes each modality to approximately [0, 1].
    Normalizes targets (t, x, y) to [0, 1] for balanced training.

    Args:
        event_ids: List of event IDs to include in this dataset.
        data_dir: Directory containing train.h5 and events.csv.
    """

    def __init__(self, event_ids: list, data_dir: str = "data"):
        self.event_ids = event_ids
        self.data_dir = data_dir
        self.h5_path = f"{data_dir}/train.h5"

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            images: (4, 36, 192, 192) normalized image tensor.
            targets: (N, 3) normalized (t, x, y) coordinates.
            event_id: str event identifier.
        """
        event_id = self.event_ids[idx]

        with h5py.File(self.h5_path, 'r') as f:
            event = {
                img_type: f[event_id][img_type][:]
                for img_type in ['vis', 'ir069', 'ir107', 'vil', 'lght']
            }

        images = self._process_images(event)
        targets = self._process_targets(event)

        return images, targets, event_id

    def _process_targets(self, event: dict) -> np.ndarray:
        """Normalize lightning targets (t, x, y) to [0, 1]."""
        original_x = event['lght'][:, 3]
        original_y = event['lght'][:, 4]
        resized_x = original_x / 2.0
        resized_y = original_y / 2.0

        t_norm = event['lght'][:, 0] / 10800.0
        x_norm = resized_x / 192.0
        y_norm = resized_y / 192.0

        targets = np.stack([t_norm, x_norm, y_norm], axis=1).astype(np.float32)

        return targets


# =============================================================================
# Density-based Dataset
# =============================================================================

class StormDensityDataset(BaseStormDataset):
    """
    PyTorch Dataset for storm prediction with density map targets.

    Loads and preprocesses storm events on-the-fly from HDF5 file.
    Creates Gaussian density maps from lightning coordinates.

    Args:
        event_ids: List of event IDs to include in this dataset.
        data_dir: Directory containing train.h5 and events.csv.
        sigma: Gaussian kernel standard deviation for density maps (default: 3.0).
        include_time: If True, returns (2, 36, 192, 192) with density and time maps.
                      If False, returns (36, 192, 192) density only (default: False).
    """

    def __init__(self, event_ids: list, data_dir: str = "data", sigma: float = 3.0, include_time: bool = False):
        self.event_ids = event_ids
        self.data_dir = data_dir
        self.h5_path = f"{data_dir}/train.h5"
        self.sigma = sigma
        self.include_time = include_time

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            images: (4, 36, 192, 192) normalized image tensor.
            target: (36, 192, 192) density map if include_time=False,
                    (2, 36, 192, 192) density+time maps if include_time=True.
            event_id: str event identifier.
        """
        event_id = self.event_ids[idx]

        with h5py.File(self.h5_path, 'r') as f:
            event = {
                img_type: f[event_id][img_type][:]
                for img_type in ['vis', 'ir069', 'ir107', 'vil', 'lght']
            }

        images = self._process_images(event)
        target = self._create_density_map(event)

        return images, target, event_id

    def _create_density_map(self, event: dict) -> np.ndarray:
        """Create density map (and optionally time map) from lightning coordinates."""
        lightning_coords = np.stack([
            event['lght'][:, 0],
            event['lght'][:, 3] / 2.0,
            event['lght'][:, 4] / 2.0
        ], axis=1).astype(np.float32)

        num_frames = 36
        height = 192
        width = 192
        sigma = float(self.sigma)
        frame_duration = 300.0

        density = np.zeros((num_frames, height, width), dtype=np.float32)

        if self.include_time:
            time_weighted_sum = np.zeros((num_frames, height, width), dtype=np.float32)
            weight_sum = np.zeros((num_frames, height, width), dtype=np.float32)

        for i in range(len(lightning_coords)):
            t_sec, x, y = lightning_coords[i]
            frame_idx = int(t_sec / frame_duration)
            frame_idx = np.clip(frame_idx, 0, num_frames - 1)

            x_int, y_int = int(round(x)), int(round(y))
            window = int(4 * sigma)
            x_min, x_max = max(0, x_int - window), min(width, x_int + window + 1)
            y_min, y_max = max(0, y_int - window), min(height, y_int + window + 1)

            if x_min >= x_max or y_min >= y_max:
                continue

            x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))

            density[frame_idx, y_min:y_max, x_min:x_max] += gaussian

            if self.include_time:
                t_in_frame = (t_sec - frame_idx * frame_duration) / frame_duration
                t_in_frame = np.clip(t_in_frame, 0.0, 1.0)
                time_weighted_sum[frame_idx, y_min:y_max, x_min:x_max] += gaussian * t_in_frame
                weight_sum[frame_idx, y_min:y_max, x_min:x_max] += gaussian

        density = np.clip(density, 0, 1)

        if self.include_time:
            time_map = np.zeros_like(density)
            mask = weight_sum > 0
            time_map[mask] = time_weighted_sum[mask] / weight_sum[mask]
            return np.stack([density, time_map], axis=0).astype(np.float32)

        return density


# =============================================================================
# Dataset Creation Functions
# =============================================================================

def create_datasets(
    data_dir: str = "data",
    test_size: float = 0.2,
    val_size: float = 0.1,
    subset_size: int = None,
    random_seed: int = 42,
) -> tuple:
    """
    Create train/validation/test StormDataset instances (coordinate-based).

    Args:
        data_dir: Directory containing train.h5 and events.csv.
        test_size: Fraction of data for test set (default: 0.2).
        val_size: Fraction of remaining data for validation (default: 0.1).
        subset_size: If provided, only use this many events (for debugging).
        random_seed: Random seed for reproducibility (default: 42).

    Returns:
        train_dataset, val_dataset, test_dataset: StormDataset instances.
    """
    df = pd.read_csv(f"{data_dir}/events.csv")
    all_ids = df['id'].unique().tolist()

    if subset_size:
        random.seed(random_seed)
        all_ids = random.sample(all_ids, subset_size)

    train_val_ids, test_ids = train_test_split(all_ids, test_size=test_size, random_state=random_seed)
    val_size_adjusted = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_adjusted, random_state=random_seed)

    train_dataset = StormDataset(train_ids, data_dir)
    val_dataset = StormDataset(val_ids, data_dir)
    test_dataset = StormDataset(test_ids, data_dir)

    return train_dataset, val_dataset, test_dataset


def create_density_datasets(
    data_dir: str = "data",
    test_size: float = 0.2,
    val_size: float = 0.1,
    subset_size: int = None,
    random_seed: int = 42,
    sigma: float = 3.0,
    include_time: bool = False,
) -> tuple:
    """
    Create train/validation/test StormDensityDataset instances.

    Args:
        data_dir: Directory containing train.h5 and events.csv.
        test_size: Fraction of data for test set (default: 0.2).
        val_size: Fraction of remaining data for validation (default: 0.1).
        subset_size: If provided, only use this many events (for debugging).
        random_seed: Random seed for reproducibility (default: 42).
        sigma: Gaussian kernel standard deviation for density maps (default: 3.0).
        include_time: If True, target is (2, 36, 192, 192) with density and time maps.
                      If False, target is (36, 192, 192) density only (default: False).

    Returns:
        train_dataset, val_dataset, test_dataset: StormDensityDataset instances.
    """
    df = pd.read_csv(f"{data_dir}/events.csv")
    all_ids = df['id'].unique().tolist()

    if subset_size:
        random.seed(random_seed)
        all_ids = random.sample(all_ids, subset_size)

    train_val_ids, test_ids = train_test_split(all_ids, test_size=test_size, random_state=random_seed)
    val_size_adjusted = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size_adjusted, random_state=random_seed)

    train_dataset = StormDensityDataset(train_ids, data_dir, sigma=sigma, include_time=include_time)
    val_dataset = StormDensityDataset(val_ids, data_dir, sigma=sigma, include_time=include_time)
    test_dataset = StormDensityDataset(test_ids, data_dir, sigma=sigma, include_time=include_time)

    return train_dataset, val_dataset, test_dataset


# =============================================================================
# Collate Functions
# =============================================================================

def _collate_fn(batch: list, max_strikes: int = 50000) -> tuple:
    """
    Collate function for coordinate-based targets (variable-length sequences).

    Pads sequences to the maximum length in the batch and creates masks
    to distinguish real strikes from padding.

    Args:
        batch: List of (images, targets, event_id) tuples from StormDataset.
        max_strikes: Maximum number of strikes to keep per sample.

    Returns:
        images: (B, 4, 36, 192, 192) stacked image tensors.
        targets_padded: (B, max_len, 3) padded target sequences.
        target_mask: (B, max_len) True for real strikes, False for padding.
        strike_counts: (B,) actual number of strikes per sample.
        ids: tuple of event IDs.
    """
    images, targets, ids = zip(*batch)

    strike_counts = [min(t.shape[0], max_strikes) for t in targets]
    max_len = max(strike_counts) if strike_counts else 1
    max_len = max(max_len, 1)

    B = len(targets)
    targets_padded = torch.zeros(B, max_len, 3)
    target_mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (t, count) in enumerate(zip(targets, strike_counts)):
        if count > 0:
            targets_padded[i, :count] = torch.tensor(t[:count])
            target_mask[i, :count] = True

    images = torch.stack([torch.tensor(img) for img in images])
    strike_counts = torch.tensor(strike_counts, dtype=torch.int32)

    return images, targets_padded, target_mask, strike_counts, ids


def _collate_fn_density(batch: list) -> tuple:
    """
    Collate function for density map targets.

    Args:
        batch: List of (images, density_map, event_id) tuples from StormDensityDataset.

    Returns:
        images: (B, 4, 36, 192, 192) stacked image tensors (float32).
        density_maps: (B, 36, 192, 192) or (B, 2, 36, 192, 192) stacked density map tensors.
        ids: tuple of event IDs.
    """
    images, density_maps, ids = zip(*batch)

    images = torch.stack([torch.tensor(img, dtype=torch.float32) for img in images])
    density_maps = torch.stack([torch.tensor(dm, dtype=torch.float32) for dm in density_maps])

    return images, density_maps, ids


# =============================================================================
# DataLoader Creation Functions
# =============================================================================

def create_dataloaders(
    train_dataset: StormDataset,
    val_dataset: StormDataset,
    test_dataset: StormDataset,
    batch_size: int = 4,
    max_strikes: int = 50000,
    num_workers: int = 0,
) -> tuple:
    """
    Create DataLoaders from coordinate-based StormDataset instances.

    Args:
        train_dataset: StormDataset for training.
        val_dataset: StormDataset for validation.
        test_dataset: StormDataset for testing.
        batch_size: Batch size for all loaders (default: 4).
        max_strikes: Maximum strikes per sample (default: 50000).
        num_workers: Number of worker processes for data loading (default: 0).

    Returns:
        train_loader, val_loader, test_loader: DataLoader instances.
    """
    collate = lambda b: _collate_fn(b, max_strikes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def create_density_dataloaders(
    train_dataset: StormDensityDataset,
    val_dataset: StormDensityDataset,
    test_dataset: StormDensityDataset,
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple:
    """
    Create DataLoaders from density-based StormDensityDataset instances.

    Works with both density-only (36, 192, 192) and density+time (2, 36, 192, 192) targets.

    Args:
        train_dataset: StormDensityDataset for training.
        val_dataset: StormDensityDataset for validation.
        test_dataset: StormDensityDataset for testing.
        batch_size: Batch size for all loaders (default: 4).
        num_workers: Number of worker processes for data loading (default: 0).
        pin_memory: Pin memory for faster GPU transfer (default: True).

    Returns:
        train_loader, val_loader, test_loader: DataLoader instances.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn_density,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn_density,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn_density,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Backward Compatibility
# =============================================================================

create_density_loaders_from_datasets = create_density_dataloaders


# =============================================================================
# Helper Functions
# =============================================================================

def load_event(event_id: str, data_dir: str = "data") -> dict:
    """
    Load all image arrays for a given event id.

    Args:
        event_id: Event identifier string.
        data_dir: Directory containing train.h5.

    Returns:
        dict with keys: 'vis', 'ir069', 'ir107', 'vil', 'lght'.
    """
    with h5py.File(f'{data_dir}/train.h5', 'r') as f:
        event = {img_type: f[event_id][img_type][:] for img_type in ['vis', 'ir069', 'ir107', 'vil', 'lght']}
    return event


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "StormDataset",
    "StormDensityDataset",
    "create_datasets",
    "create_density_datasets",
    "create_dataloaders",
    "create_density_dataloaders",
    "create_density_loaders_from_datasets",
    "load_event",
]
