"""
Task 4 Training Utilities for Lightning Prediction.

This module contains loss functions, training loops, and utilities for training
the lightning prediction models defined in task4.model.

The training functions are exact copies from the task4 Jupyter notebooks to ensure
consistency between notebook experimentation and production code.

Usage:
    from task4.train import focal_loss, train_density_map_predictor
    from task4.model import DensityMapPredictor

    model = DensityMapPredictor()
    train_density_map_predictor(model, train_loader, val_loader, optimizer, scheduler, device)

Available Loss Functions:
    - focal_loss: Focal loss for imbalanced density map prediction
    - combined_loss: Focal loss on density + L1 on time
    - dual_decoder_loss: Loss for dual-decoder models (density + time)
    - masked_coordinate_loss: Masked loss for direct coordinate prediction
    - chamfer_distance_loss: Order-invariant loss for set prediction
    - compute_dual_decoder_loss: Loss for DualCNNLightningPredictor
    - compute_dual_decoder_chamfer_loss: Chamfer + stop loss for DualCNNLightningPredictor
    - poisson_loss: Poisson NLL for count prediction
    - weighted_poisson_loss: Weighted Poisson loss emphasizing lightning frames

Utility Functions:
    - create_density_map: Convert (t,x,y) coordinates to density maps
    - create_density_and_time_map: Create combined density + time targets
    - density_to_coordinates: Extract coordinates from density predictions
    - density_time_to_coordinates: Extract coords from density+time predictions
    - compute_iou: Compute IoU between predicted and target density maps
    - compute_spatial_error: Compute mean spatial error between coordinates
    - compute_temporal_error: Compute mean temporal error between coordinates

Training Functions (exact copies from notebooks):
    From task4.ipynb (CNNLightningPredictor):
        - train_cnn_lightning_predictor: Full training loop with masked L1 loss

    From task4_density_probabilies.ipynb (DensityMapPredictor):
        - train_density_map_predictor: Full training loop with focal loss

    From task4_transformer.ipynb (DualCNNLightningPredictor):
        - train_dual_cnn_lightning_predictor: Full training loop with masked/Chamfer loss

    From task4-time_pred.ipynb (TimeLightningModel):
        - train_time_lightning_epoch: Single epoch with weighted Poisson loss
        - validate_time_lightning: Validation function
        - train_time_lightning_model: Full training loop

    From task4_probability_time.ipynb (DensityTimePredictor/UNet):
        - train_density_time_predictor: Full training loop with combined/dual_decoder loss

    From task4-density_time_pred.ipynb (LightningTimePredictor):
        - train_lightning_time_epoch: Single epoch with MSE + smoothness loss
        - validate_lightning_time: Validation function
        - train_lightning_time_predictor: Full training loop
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter


# =============================================================================
# Loss Functions
# =============================================================================

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for imbalanced binary classification.

    Down-weights easy negatives (empty pixels) to focus on hard cases.
    Essential for density map prediction where most pixels are empty.

    Args:
        pred: (B, T, H, W) or (B, H, W) predicted probabilities in [0, 1]
        target: Same shape as pred, ground truth density maps
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter - higher values focus more on hard examples (default: 2.0)

    Returns:
        Scalar focal loss value
    """
    pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

    bce = F.binary_cross_entropy(pred, target, reduction='none')

    # pt = p if y=1, (1-p) if y=0
    pt = torch.where(target > 0.5, pred, 1 - pred)

    # Focal weight: (1 - pt)^gamma
    focal_weight = (1 - pt) ** gamma

    # Alpha weighting for positive/negative
    alpha_weight = torch.where(target > 0.5, alpha, 1 - alpha)

    loss = alpha_weight * focal_weight * bce
    return loss.mean()


def combined_loss(pred, target, alpha=0.25, gamma=2.0, time_weight=1.0):
    """
    Combined loss: focal loss on density + masked L1 on time.

    For models that output (B, 2, T, H, W) where:
    - Channel 0: density probability
    - Channel 1: time within frame [0, 1]

    Args:
        pred: (B, 2, 36, H, W) - channel 0 = density, channel 1 = time
        target: (B, 2, 36, H, W) - same format as pred
        alpha: Focal loss alpha
        gamma: Focal loss gamma
        time_weight: Weight for time loss relative to density loss

    Returns:
        total_loss, density_loss, time_loss
    """
    pred_density = pred[:, 0]    # (B, 36, H, W)
    pred_time = pred[:, 1]       # (B, 36, H, W)
    gt_density = target[:, 0]    # (B, 36, H, W)
    gt_time = target[:, 1]       # (B, 36, H, W)

    # Focal loss on density channel
    density_loss = focal_loss(pred_density, gt_density, alpha, gamma)

    # Masked L1 loss on time channel (only where lightning exists)
    mask = (gt_density > 0.5).float()
    if mask.sum() > 0:
        time_loss = (mask * (pred_time - gt_time).abs()).sum() / mask.sum()
    else:
        time_loss = torch.tensor(0.0, device=pred.device)

    total_loss = density_loss + time_weight * time_loss
    return total_loss, density_loss, time_loss


def dual_decoder_loss(pred, target, alpha=0.25, gamma=2.0, time_weight=5.0):
    """
    Combined loss for dual-decoder UNet: focal loss on density + masked smooth L1 on time.

    Args:
        pred: (B, 2, 36, H, W) - channel 0 = density, channel 1 = time
        target: (B, 2, 36, H, W) - same format
        alpha: Focal loss alpha for class imbalance
        gamma: Focal loss gamma for hard example mining
        time_weight: Weight for time loss (higher = more emphasis on time)

    Returns:
        total_loss, density_loss, time_loss
    """
    pred_density = pred[:, 0]
    pred_time = pred[:, 1]
    gt_density = target[:, 0]
    gt_time = target[:, 1]

    # Focal loss on density
    density_loss = focal_loss(pred_density, gt_density, alpha, gamma)

    # Masked Smooth L1 loss on time (better gradients near zero)
    mask = (gt_density > 0.5).float()
    if mask.sum() > 0:
        time_diff = F.smooth_l1_loss(pred_time, gt_time, reduction='none')
        time_loss = (mask * time_diff).sum() / mask.sum()
    else:
        time_loss = torch.tensor(0.0, device=pred.device)

    total_loss = density_loss + time_weight * time_loss
    return total_loss, density_loss, time_loss


def masked_coordinate_loss(predictions, targets, event_counts, criterion=None):
    """
    Masked loss for direct coordinate prediction models.

    Only computes loss on valid events (not padding).

    Args:
        predictions: (B, max_events, 3) predicted (t, x, y)
        targets: (B, max_events, 3) target (t, x, y)
        event_counts: (B,) number of actual events per sample
        criterion: Loss function (default: L1Loss)

    Returns:
        Scalar loss value
    """
    if criterion is None:
        criterion = nn.L1Loss()

    device = predictions.device
    batch_loss = torch.tensor(0.0, device=device)

    for i in range(predictions.size(0)):
        actual_num_events = int(event_counts[i].item())
        if actual_num_events > 0:
            batch_loss += criterion(
                predictions[i, :actual_num_events],
                targets[i, :actual_num_events]
            )

    return batch_loss


def chamfer_distance_loss(pred_coords, gt_coords, pred_mask, gt_mask,
                          lambda_spatial=1.0, lambda_time=2.0):
    """
    Chamfer distance loss for set-based supervision.

    Order-invariant loss that finds optimal assignment between predictions
    and ground truth. Does not require ordering alignment.

    Args:
        pred_coords: (B, N_pred, 3) - predicted (t, x, y) normalized to [0, 1]
        gt_coords: (B, N_gt, 3) - ground truth (t, x, y)
        pred_mask: (B, N_pred) - True for valid predictions
        gt_mask: (B, N_gt) - True for valid ground truth
        lambda_spatial: Weight for spatial distance
        lambda_time: Weight for temporal distance

    Returns:
        Chamfer distance loss (scalar)
    """
    B = pred_coords.shape[0]
    device = pred_coords.device
    total_loss = torch.tensor(0.0, device=device)
    valid_batches = 0

    for b in range(B):
        pred_valid = pred_coords[b, pred_mask[b]]  # (M, 3)
        gt_valid = gt_coords[b, gt_mask[b]]        # (K, 3)

        M, K = len(pred_valid), len(gt_valid)

        if M == 0 or K == 0:
            if M > 0 and K == 0:
                total_loss = total_loss + 1.0  # Penalty for false positives
                valid_batches += 1
            continue

        # Separate spatial and temporal components
        pred_time = pred_valid[:, 0:1]     # (M, 1)
        pred_spatial = pred_valid[:, 1:3]  # (M, 2)
        gt_time = gt_valid[:, 0:1]         # (K, 1)
        gt_spatial = gt_valid[:, 1:3]      # (K, 2)

        # Compute pairwise distances
        dist_spatial = torch.cdist(pred_spatial, gt_spatial, p=2)  # (M, K)
        dist_time = torch.cdist(pred_time, gt_time, p=1)           # (M, K)

        # Combined weighted distance
        dist = lambda_spatial * dist_spatial + lambda_time * dist_time

        # Chamfer distance: pred→gt + gt→pred
        loss_p2g = dist.min(dim=1).values.mean()
        loss_g2p = dist.min(dim=0).values.mean()

        total_loss = total_loss + loss_p2g + loss_g2p
        valid_batches += 1

    if valid_batches == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / valid_batches


def compute_dual_decoder_loss(spatial_preds, temporal_preds, stop_preds,
                               targets, mask, stop_weight=1.0):
    """
    Loss for DualCNNLightningPredictor model.

    Args:
        spatial_preds: (B, N, 2) predicted (x, y)
        temporal_preds: (B, N, 1) predicted t
        stop_preds: (B, N) stop logits
        targets: (B, N, 3) target (t, x, y)
        mask: (B, N) True for real strikes, False for padding
        stop_weight: Weight for stop loss

    Returns:
        total_loss, coord_loss (float), stop_loss (float)
    """
    B, N, _ = targets.shape

    # Combine predictions: (t, x, y)
    strike_preds = torch.cat([temporal_preds, spatial_preds], dim=-1)

    # Regression loss on real strikes only (Smooth L1)
    coord_loss = F.smooth_l1_loss(strike_preds, targets, reduction='none')
    coord_loss = coord_loss.sum(dim=-1)  # (B, N)
    coord_loss = (coord_loss * mask.float()).sum() / (mask.sum() + 1e-8)

    # Stop token loss
    stop_targets = torch.zeros_like(stop_preds)
    for i in range(B):
        real_count = mask[i].sum().int()
        if real_count < N:
            stop_targets[i, real_count] = 1.0

    stop_mask = torch.zeros_like(stop_preds, dtype=torch.bool)
    for i in range(B):
        real_count = mask[i].sum().int()
        stop_mask[i, :min(real_count + 1, N)] = True

    stop_loss = F.binary_cross_entropy_with_logits(
        stop_preds, stop_targets, reduction='none'
    )
    stop_loss = (stop_loss * stop_mask.float()).sum() / (stop_mask.sum() + 1e-8)

    total_loss = coord_loss + stop_weight * stop_loss

    return total_loss, coord_loss.item(), stop_loss.item()


def compute_dual_decoder_chamfer_loss(spatial_preds, temporal_preds, stop_preds,
                                       targets, mask, stop_weight=1.0,
                                       lambda_spatial=1.0, lambda_time=2.0):
    """
    Combined loss using Chamfer distance for DualCNNLightningPredictor.

    Args:
        spatial_preds: (B, N, 2) predicted (x, y)
        temporal_preds: (B, N, 1) predicted t
        stop_preds: (B, N) stop logits
        targets: (B, N, 3) target (t, x, y)
        mask: (B, N) True for real strikes
        stop_weight: Weight for stop loss
        lambda_spatial: Weight for spatial distance
        lambda_time: Weight for temporal distance

    Returns:
        total_loss, coord_loss, stop_loss
    """
    B, N, _ = targets.shape

    pred_coords = torch.cat([temporal_preds, spatial_preds], dim=-1)

    stop_probs = torch.sigmoid(stop_preds)
    pred_mask = stop_probs < 0.5

    # Ensure enough active predictions
    for b in range(B):
        gt_count = mask[b].sum().int()
        pred_mask[b, :gt_count] = True

    # Chamfer loss
    coord_loss = chamfer_distance_loss(
        pred_coords, targets, pred_mask, mask,
        lambda_spatial=lambda_spatial, lambda_time=lambda_time
    )

    # Stop token loss
    stop_targets = torch.zeros_like(stop_preds)
    for i in range(B):
        real_count = mask[i].sum().int()
        if real_count < N:
            stop_targets[i, real_count] = 1.0

    stop_mask = torch.zeros_like(stop_preds, dtype=torch.bool)
    for i in range(B):
        real_count = mask[i].sum().int()
        stop_mask[i, :min(real_count + 1, N)] = True

    stop_loss = F.binary_cross_entropy_with_logits(
        stop_preds, stop_targets, reduction='none'
    )
    stop_loss = (stop_loss * stop_mask.float()).sum() / (stop_mask.sum() + 1e-8)

    total_loss = coord_loss + stop_weight * stop_loss

    coord_loss_val = coord_loss.item() if isinstance(coord_loss, torch.Tensor) else coord_loss
    return total_loss, coord_loss_val, stop_loss.item()


def poisson_loss(pred, target, log_input=True):
    """
    Poisson negative log-likelihood loss for count prediction.

    Args:
        pred: (B, T) predicted values (log-counts if log_input=True)
        target: (B, T) ground truth counts
        log_input: If True, pred is log(lambda), else pred is lambda

    Returns:
        Scalar loss value
    """
    if log_input:
        # pred is log(lambda)
        loss = torch.exp(pred) - target * pred
    else:
        # pred is lambda
        pred = torch.clamp(pred, min=1e-8)
        loss = pred - target * torch.log(pred)

    return loss.mean()


def weighted_poisson_loss(pred, target, lightning_weight=10.0, log_input=True):
    """
    Weighted Poisson loss that emphasizes frames with lightning.

    Args:
        pred: (B, T) predicted values
        target: (B, T) ground truth counts
        lightning_weight: Weight for frames with lightning
        log_input: If True, pred is log(lambda)

    Returns:
        Scalar loss value
    """
    weights = torch.ones_like(target)
    weights[target > 0] = lightning_weight

    if log_input:
        loss = (torch.exp(pred) - target * pred) * weights
    else:
        pred = torch.clamp(pred, min=1e-8)
        loss = (pred - target * torch.log(pred)) * weights

    return loss.mean()


# =============================================================================
# Target Creation Utilities
# =============================================================================

def create_density_map(lightning_events, height=192, width=192, num_frames=36, sigma=3.0):
    """
    Convert lightning (t, x, y) coordinates to Gaussian density maps.

    Args:
        lightning_events: (N, 3) array of [time_seconds, x, y]
        height: Height of output density map
        width: Width of output density map
        num_frames: Number of temporal frames (36 = 3 hours at 5 min each)
        sigma: Gaussian kernel width (larger = more spread)

    Returns:
        density: (num_frames, height, width) array with values in [0, 1]
    """
    density = np.zeros((num_frames, height, width), dtype=np.float32)
    frame_duration = 300.0  # 5 minutes per frame

    for i in range(len(lightning_events)):
        t_sec, x, y = lightning_events[i]
        frame_idx = int(t_sec / frame_duration)
        frame_idx = np.clip(frame_idx, 0, num_frames - 1)

        x_int, y_int = int(round(x)), int(round(y))

        # Compute in local window for efficiency
        window = int(4 * sigma)
        x_min, x_max = max(0, x_int - window), min(width, x_int + window + 1)
        y_min, y_max = max(0, y_int - window), min(height, y_int + window + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        x_grid, y_grid = np.meshgrid(
            np.arange(x_min, x_max),
            np.arange(y_min, y_max)
        )
        gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))

        density[frame_idx, y_min:y_max, x_min:x_max] += gaussian

    return np.clip(density, 0, 1)


def create_density_and_time_map(lightning_events, height=192, width=192,
                                 num_frames=36, sigma=3.0):
    """
    Convert lightning (t, x, y) to per-frame density + time maps.

    Channel 0: P(x,y) per frame - Gaussian density of strikes
    Channel 1: t(x,y) per frame - Gaussian-weighted mean time within frame [0,1]

    Args:
        lightning_events: (N, 3) array of [time_seconds, x, y]
        height: Height of output map
        width: Width of output map
        num_frames: Number of temporal frames
        sigma: Gaussian kernel width

    Returns:
        target: (2, num_frames, height, width) float32 array
    """
    density = np.zeros((num_frames, height, width), dtype=np.float32)
    time_weighted_sum = np.zeros((num_frames, height, width), dtype=np.float32)
    weight_sum = np.zeros((num_frames, height, width), dtype=np.float32)

    frame_duration = 300.0  # 5 minutes per frame

    for i in range(len(lightning_events)):
        t_sec, x, y = lightning_events[i]
        frame_idx = int(t_sec / frame_duration)
        frame_idx = np.clip(frame_idx, 0, num_frames - 1)

        # Normalize time within frame to [0, 1]
        t_in_frame = (t_sec - frame_idx * frame_duration) / frame_duration
        t_in_frame = np.clip(t_in_frame, 0.0, 1.0)

        x_int, y_int = int(round(x)), int(round(y))

        window = int(4 * sigma)
        x_min, x_max = max(0, x_int - window), min(width, x_int + window + 1)
        y_min, y_max = max(0, y_int - window), min(height, y_int + window + 1)

        if x_min >= x_max or y_min >= y_max:
            continue

        x_grid, y_grid = np.meshgrid(
            np.arange(x_min, x_max),
            np.arange(y_min, y_max)
        )
        gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))

        density[frame_idx, y_min:y_max, x_min:x_max] += gaussian
        time_weighted_sum[frame_idx, y_min:y_max, x_min:x_max] += gaussian * t_in_frame
        weight_sum[frame_idx, y_min:y_max, x_min:x_max] += gaussian

    # Clip density to [0, 1]
    density = np.clip(density, 0, 1)

    # Compute weighted average time within frame
    time_map = np.zeros_like(density)
    mask = weight_sum > 0
    time_map[mask] = time_weighted_sum[mask] / weight_sum[mask]

    return np.stack([density, time_map], axis=0).astype(np.float32)


def density_to_coordinates(density_map, threshold=0.3, min_distance=5):
    """
    Extract (t, x, y) coordinates from predicted density map.

    Uses local maximum detection to find lightning locations.

    Args:
        density_map: (T, H, W) array of probabilities
        threshold: Minimum probability to consider as lightning
        min_distance: Minimum pixels between detected peaks

    Returns:
        predictions: (N, 3) array of [time_seconds, x, y]
    """
    predictions = []
    frame_duration = 300.0  # 5 minutes per frame

    for frame_idx in range(density_map.shape[0]):
        frame = density_map[frame_idx]

        # Find local maxima
        local_max = maximum_filter(frame, size=min_distance)
        peaks = (frame == local_max) & (frame > threshold)

        y_coords, x_coords = np.where(peaks)
        for x, y in zip(x_coords, y_coords):
            t_seconds = frame_idx * frame_duration
            predictions.append([t_seconds, x, y])

    return np.array(predictions) if predictions else np.zeros((0, 3))


def density_time_to_coordinates(density_time_map, threshold=0.3, min_distance=5):
    """
    Extract (t, x, y) coordinates from per-frame density+time prediction.

    Args:
        density_time_map: (2, 36, H, W) - channel 0 = density, channel 1 = time [0,1]
        threshold: Minimum density to consider as lightning
        min_distance: Minimum pixels between detected peaks

    Returns:
        predictions: (N, 3) array of [time_seconds, x, y]
    """
    num_frames = density_time_map.shape[1]
    frame_duration = 300.0
    predictions = []

    for f in range(num_frames):
        density = density_time_map[0, f]
        time_map = density_time_map[1, f]

        local_max = maximum_filter(density, size=min_distance)
        peaks = (density == local_max) & (density > threshold)

        y_coords, x_coords = np.where(peaks)
        for x, y in zip(x_coords, y_coords):
            # Convert frame index + sub-frame time to absolute seconds
            t_seconds = f * frame_duration + time_map[y, x] * frame_duration
            predictions.append([t_seconds, x, y])

    return np.array(predictions) if predictions else np.zeros((0, 3))

# =============================================================================
# Training Functions - Exact copies from task4 notebooks
# =============================================================================

# -----------------------------------------------------------------------------
# From task4.ipynb - Training loop for CNNLightningPredictor
# Uses masked MSE/L1 loss for direct coordinate prediction
# -----------------------------------------------------------------------------

def train_cnn_lightning_predictor(model, train_loader, val_loader, optimizer, criterion,
                                   device, num_epochs=25):
    """
    Training loop for CNNLightningPredictor model.
    From: notebooks-training/task4.ipynb

    Args:
        model: CNNLightningPredictor model
        train_loader: DataLoader yielding (images, targets, event_counts, ids)
        val_loader: Validation DataLoader
        optimizer: Optimizer
        criterion: Loss function (e.g., nn.L1Loss())
        device: torch device
        num_epochs: Number of training epochs

    Returns:
        train_losses, val_losses: Lists of losses per epoch
    """
    from IPython.display import clear_output
    import matplotlib.pyplot as plt

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        model.train()

        for batch_idx, (images, targets, event_counts, ids) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            event_counts = event_counts.to(device)

            optimizer.zero_grad()
            predictions = model(images)

            # Calculate masked MSE loss
            batch_loss = torch.tensor(0.0, device=device)
            for i in range(images.size(0)):
                actual_num_events = event_counts[i].item()
                if actual_num_events > 0:
                    batch_loss += criterion(predictions[i, :actual_num_events], targets[i, :actual_num_events])

            if batch_loss.item() > 0:
                batch_loss.backward()
                optimizer.step()

            total_train_loss += batch_loss.item()

        epoch_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets, event_counts, ids in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                event_counts = event_counts.to(device)

                predictions = model(images)

                batch_loss = torch.tensor(0.0, device=device)
                for i in range(images.size(0)):
                    actual_num_events = event_counts[i].item()
                    if actual_num_events > 0:
                        batch_loss += criterion(predictions[i, :actual_num_events], targets[i, :actual_num_events])

                total_val_loss += batch_loss.item()

        epoch_val_loss = total_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # Update plot after each epoch
        clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), train_losses, marker='o', label='Training Loss')
        plt.plot(range(1, epoch + 2), val_losses, marker='s', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress - Epoch {epoch + 1}/{num_epochs}')
        plt.legend()
        plt.grid(True)
        plt.xlim(0.5, num_epochs + 0.5)
        plt.show()

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    print("Training complete.")
    return train_losses, val_losses


# -----------------------------------------------------------------------------
# From task4_density_probabilies.ipynb - Training loop for DensityMapPredictor
# Uses focal loss for density map prediction
# -----------------------------------------------------------------------------

def train_density_map_predictor(model, train_loader, val_loader, optimizer, scheduler,
                                 device, num_epochs=30):
    """
    Training loop for DensityMapPredictor model.
    From: notebooks-training/task4_density_probabilies.ipynb

    Args:
        model: DensityMapPredictor model
        train_loader: DataLoader yielding (images, density_targets, ids)
        val_loader: Validation DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (ReduceLROnPlateau)
        device: torch device
        num_epochs: Number of training epochs

    Returns:
        train_losses, val_losses: Lists of losses per epoch
    """
    from IPython.display import clear_output
    import matplotlib.pyplot as plt

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0

        for batch_idx, (images, density_targets, ids) in enumerate(train_loader):
            images = images.to(device)
            density_targets = density_targets.to(device)

            optimizer.zero_grad()
            pred_density = model(images)

            loss = focal_loss(pred_density, density_targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, density_targets, ids in val_loader:
                images = images.to(device)
                density_targets = density_targets.to(device)

                pred_density = model(images)
                loss = focal_loss(pred_density, density_targets)
                total_val_loss += loss.item()

        epoch_val_loss = total_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # Learning rate scheduling
        scheduler.step(epoch_val_loss)

        # Update plot after each epoch
        clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), train_losses, marker='o', label='Training Loss')
        plt.plot(range(1, epoch + 2), val_losses, marker='s', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Focal Loss')
        plt.title(f'Training Progress - Epoch {epoch + 1}/{num_epochs}')
        plt.legend()
        plt.grid(True)
        plt.xlim(0.5, num_epochs + 0.5)
        plt.show()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

    print("Training complete.")
    return train_losses, val_losses


# -----------------------------------------------------------------------------
# From task4_transformer.ipynb - Training loop for DualCNNLightningPredictor
# Uses masked loss or Chamfer distance loss
# -----------------------------------------------------------------------------

def train_dual_cnn_lightning_predictor(dual_model, train_loader, val_loader, optimizer,
                                        scheduler, device, num_epochs=20,
                                        use_chamfer_loss=False):
    """
    Training loop for DualCNNLightningPredictor model.
    From: notebooks-training/task4_transformer.ipynb

    Args:
        dual_model: DualCNNLightningPredictor model
        train_loader: DataLoader yielding (images, targets, mask, strike_counts, ids)
        val_loader: Validation DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (CosineAnnealingLR)
        device: torch device
        num_epochs: Number of training epochs
        use_chamfer_loss: If True, use Chamfer distance loss; else use masked loss

    Returns:
        train_losses, val_losses, train_coord_losses, train_stop_losses
    """
    from IPython.display import clear_output
    import matplotlib.pyplot as plt

    train_losses = []
    val_losses = []
    train_coord_losses = []
    train_stop_losses = []
    best_val_loss = float('inf')

    loss_name = "Chamfer" if use_chamfer_loss else "Masked"
    print(f"Training Dual-CNN with {loss_name} loss")

    for epoch in range(num_epochs):
        # Training
        dual_model.train()
        total_train_loss = 0.0
        total_coord_loss = 0.0
        total_stop_loss = 0.0

        for batch_idx, (images, targets, mask, strike_counts, ids) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            spatial_preds, temporal_preds, stop_preds = dual_model(images, targets)

            if use_chamfer_loss:
                loss, c_loss, s_loss = compute_dual_decoder_chamfer_loss(
                    spatial_preds, temporal_preds, stop_preds, targets, mask
                )
            else:
                loss, c_loss, s_loss = compute_dual_decoder_loss(
                    spatial_preds, temporal_preds, stop_preds, targets, mask
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(dual_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_coord_loss += c_loss
            total_stop_loss += s_loss

        n_batches = len(train_loader)
        avg_train_loss = total_train_loss / n_batches
        train_losses.append(avg_train_loss)
        train_coord_losses.append(total_coord_loss / n_batches)
        train_stop_losses.append(total_stop_loss / n_batches)

        # Validation
        dual_model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, targets, mask, strike_counts, ids in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                mask = mask.to(device)

                spatial_preds, temporal_preds, stop_preds = dual_model(images, targets)

                if use_chamfer_loss:
                    loss, _, _ = compute_dual_decoder_chamfer_loss(
                        spatial_preds, temporal_preds, stop_preds, targets, mask
                    )
                else:
                    loss, _, _ = compute_dual_decoder_loss(
                        spatial_preds, temporal_preds, stop_preds, targets, mask
                    )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step()

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(dual_model.state_dict(), 'best_dual_cnn_predictor.pth')

        # Plot progress
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs_range = range(1, epoch + 2)

        axes[0].plot(epochs_range, train_losses, marker='o', label='Train')
        axes[0].plot(epochs_range, val_losses, marker='s', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title(f'Dual-CNN Total Loss ({loss_name})')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs_range, train_coord_losses, marker='o', label='Coord Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Coordinate Loss')
        axes[1].set_title('Coordinate Loss')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(epochs_range, train_stop_losses, marker='o', label='Stop Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Stop Loss')
        axes[2].set_title('Stop Token Loss')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

        print(f"Epoch [{epoch+1}/{num_epochs}] (Dual-CNN with {loss_name} Loss)")
        print(f"  Train: {avg_train_loss:.4f} (Coord: {train_coord_losses[-1]:.4f}, Stop: {train_stop_losses[-1]:.4f})")
        print(f"  Val:   {avg_val_loss:.4f}")
        print(f"  LR:    {scheduler.get_last_lr()[0]:.6f}")

    print("\nTraining complete!")
    return train_losses, val_losses, train_coord_losses, train_stop_losses


# -----------------------------------------------------------------------------
# From task4-time_pred.ipynb - Training functions for TimeLightningModel
# Uses weighted Poisson NLL loss for count prediction
# -----------------------------------------------------------------------------

def train_time_lightning_epoch(model, loader, optimizer, device, lightning_weight=10.0):
    """
    Single training epoch for TimeLightningModel.
    From: notebooks-training/task4-time_pred.ipynb

    Args:
        model: TimeLightningModel that outputs log(lambda)
        loader: DataLoader yielding (images, counts, ids)
        optimizer: Optimizer
        device: torch device
        lightning_weight: Weight multiplier for frames with lightning

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0

    for images, counts, _ in loader:
        images = images.to(device)
        counts = counts.to(device)

        optimizer.zero_grad()

        # Model outputs log(lambda)
        outputs = model(images)  # (B, 36)

        # Weight mask: emphasize bins with lightning
        weights = torch.ones_like(counts)
        weights[counts > 0] = lightning_weight

        # Weighted Poisson NLL (without log(k!))
        # loss = lambda - k * log(lambda)
        # here: lambda = exp(outputs)
        loss = (torch.exp(outputs) - counts * outputs)
        loss = (loss * weights).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_time_lightning(model, loader, criterion, device):
    """
    Validation function for TimeLightningModel.
    From: notebooks-training/task4-time_pred.ipynb

    Args:
        model: TimeLightningModel
        loader: Validation DataLoader yielding (images, counts, ids)
        criterion: Loss function (e.g., nn.PoissonNLLLoss)
        device: torch device

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():

        for images, counts, _ in loader:
            images = images.to(device)
            counts = counts.to(device)

            outputs = model(images)

            loss = torch.exp(outputs) - counts * outputs
            loss = loss.mean(dim=1)  # per-event

            total_loss += loss.item()
    return total_loss / len(loader)


def train_time_lightning_model(model, train_dataset, valid_dataset, device,
                                seed=42, lr=1e-4, batch_size=4, val_batch_size=8,
                                n_epochs=10, lightning_weight=10.0, plot_every=1):
    """
    Full training loop for TimeLightningModel.
    From: notebooks-training/task4-time_pred.ipynb

    Args:
        model: TimeLightningModel
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        device: torch device
        seed: Random seed
        lr: Learning rate
        batch_size: Training batch size
        val_batch_size: Validation batch size
        n_epochs: Number of epochs
        lightning_weight: Weight for frames with lightning
        plot_every: Plot validation examples every N epochs

    Returns:
        train_loader, valid_loader
    """
    from torch.utils.data import DataLoader
    import random
    import numpy as np

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.PoissonNLLLoss(log_input=True)

    model.to(device)

    for epoch in range(n_epochs):
        train_loss = train_time_lightning_epoch(
            model,
            train_loader,
            optimizer,
            device,
            lightning_weight=lightning_weight,
        )

        val_loss = validate_time_lightning(
            model,
            valid_loader,
            criterion,
            device,
        )

        print(
            f"Epoch [{epoch+1}/{n_epochs}] | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

    return train_loader, valid_loader


# -----------------------------------------------------------------------------
# From task4_probability_time.ipynb - Training loop for DensityTimePredictor/UNet
# Uses combined_loss (focal + masked L1) or dual_decoder_loss
# -----------------------------------------------------------------------------

def train_density_time_predictor(model, train_loader, val_loader, optimizer, scheduler,
                                  device, num_epochs=30, loss_fn='combined'):
    """
    Training loop for DensityTimePredictor or DensityTimeUNet models.
    From: notebooks-training/task4_probability_time.ipynb

    Args:
        model: DensityTimePredictor, DensityTimeUNet, DualDecoderUNet, or DualDecoderUNetV2
        train_loader: DataLoader yielding (images, targets, ids)
        val_loader: Validation DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (ReduceLROnPlateau)
        device: torch device
        num_epochs: Number of training epochs
        loss_fn: 'combined' for combined_loss, 'dual_decoder' for dual_decoder_loss

    Returns:
        Dictionary with train_losses, val_losses, train_density_losses, etc.
    """
    from IPython.display import clear_output
    import matplotlib.pyplot as plt

    # Select loss function
    if loss_fn == 'dual_decoder':
        loss_func = dual_decoder_loss
    else:
        loss_func = combined_loss

    train_losses = []
    val_losses = []
    train_density_losses = []
    train_time_losses = []
    val_density_losses = []
    val_time_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        total_train_density = 0.0
        total_train_time = 0.0

        for batch_idx, (images, targets, ids) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred = model(images)

            loss, d_loss, t_loss = loss_func(pred, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_density += d_loss.item()
            total_train_time += t_loss.item() if isinstance(t_loss, torch.Tensor) else t_loss

        n_batches = len(train_loader)
        train_losses.append(total_train_loss / n_batches)
        train_density_losses.append(total_train_density / n_batches)
        train_time_losses.append(total_train_time / n_batches)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        total_val_density = 0.0
        total_val_time = 0.0
        with torch.no_grad():
            for images, targets, ids in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                pred = model(images)
                loss, d_loss, t_loss = loss_func(pred, targets)
                total_val_loss += loss.item()
                total_val_density += d_loss.item()
                total_val_time += t_loss.item() if isinstance(t_loss, torch.Tensor) else t_loss

        n_val = len(val_loader)
        val_losses.append(total_val_loss / n_val)
        val_density_losses.append(total_val_density / n_val)
        val_time_losses.append(total_val_time / n_val)

        scheduler.step(val_losses[-1])

        # Update plot
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs_range = range(1, epoch + 2)

        axes[0].plot(epochs_range, train_losses, marker='o', label='Train Total')
        axes[0].plot(epochs_range, val_losses, marker='s', label='Val Total')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs_range, train_density_losses, marker='o', label='Train Density')
        axes[1].plot(epochs_range, val_density_losses, marker='s', label='Val Density')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Focal Loss')
        axes[1].set_title('Density (Focal) Loss')
        axes[1].legend()
        axes[1].grid(True)

        time_loss_name = 'Masked Smooth L1' if loss_fn == 'dual_decoder' else 'Masked L1'
        axes[2].plot(epochs_range, train_time_losses, marker='o', label='Train Time')
        axes[2].plot(epochs_range, val_time_losses, marker='s', label='Val Time')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel(f'{time_loss_name} Loss')
        axes[2].set_title(f'Time ({time_loss_name}) Loss')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

        print(f"Epoch [{epoch+1}/{num_epochs}]  Total: {train_losses[-1]:.6f}/{val_losses[-1]:.6f}  "
              f"Density: {train_density_losses[-1]:.6f}/{val_density_losses[-1]:.6f}  "
              f"Time: {train_time_losses[-1]:.6f}/{val_time_losses[-1]:.6f}")

    print("Training complete.")
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_density_losses': train_density_losses,
        'train_time_losses': train_time_losses,
        'val_density_losses': val_density_losses,
        'val_time_losses': val_time_losses
    }


# -----------------------------------------------------------------------------
# From task4-density_time_pred.ipynb - Training functions for LightningTimePredictor
# Uses MSE loss with temporal smoothness regularization
# -----------------------------------------------------------------------------

def train_lightning_time_epoch(model, loader, optimizer, device, smooth_weight=0.1):
    """
    Single training epoch for LightningTimePredictor.
    From: notebooks-training/task4-density_time_pred.ipynb

    Args:
        model: LightningTimePredictor model
        loader: DataLoader yielding (positions, masks, time_counts, ids)
        optimizer: Optimizer
        device: torch device
        smooth_weight: Weight for temporal smoothness loss

    Returns:
        Average training loss
    """
    model.train()
    train_loss = 0.0

    for positions, masks, time_counts, _ in loader:
        positions = positions.to(device)
        masks = masks.to(device)
        time_counts = time_counts.to(device)

        optimizer.zero_grad()

        # Forward
        pred_counts = model(positions, masks)  # (B, num_bins)

        # Main loss (MSE)
        mse_loss = F.mse_loss(pred_counts, time_counts)

        # Temporal smoothness loss
        smooth_loss = ((pred_counts[:, 1:] - pred_counts[:, :-1]) ** 2).mean()

        # Total loss
        loss = mse_loss + smooth_weight * smooth_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * positions.size(0)

    train_loss /= len(loader.dataset)
    return train_loss


def validate_lightning_time(model, loader, device, smooth_weight=0.1):
    """
    Validation function for LightningTimePredictor.
    From: notebooks-training/task4-density_time_pred.ipynb

    Args:
        model: LightningTimePredictor model
        loader: Validation DataLoader
        device: torch device
        smooth_weight: Weight for temporal smoothness loss

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for positions, masks, time_counts, _ in loader:
            positions = positions.to(device)
            masks = masks.to(device)
            time_counts = time_counts.to(device)

            pred_counts = model(positions, masks)

            mse_loss = F.mse_loss(pred_counts, time_counts)
            smooth_loss = ((pred_counts[:, 1:] - pred_counts[:, :-1]) ** 2).mean()

            loss = mse_loss + smooth_weight * smooth_loss
            val_loss += loss.item() * positions.size(0)

    val_loss /= len(loader.dataset)
    return val_loss


def train_lightning_time_predictor(model, train_dataset, val_dataset, device,
                                    collate_fn, seed=42, lr=1e-4, batch_size=8,
                                    val_batch_size=8, n_epochs=10, plot_every=1):
    """
    Full training loop for LightningTimePredictor.
    From: notebooks-training/task4-density_time_pred.ipynb

    Args:
        model: LightningTimePredictor model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: torch device
        collate_fn: Custom collate function for the dataset
        seed: Random seed
        lr: Learning rate
        batch_size: Training batch size
        val_batch_size: Validation batch size
        n_epochs: Number of epochs
        plot_every: Plot validation examples every N epochs

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    import random
    import numpy as np

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(n_epochs):
        train_loss = train_lightning_time_epoch(model, train_loader, optimizer, device, smooth_weight=0.1)

        val_loss = validate_lightning_time(model, val_loader, device, smooth_weight=0.1)

        print(
            f"Epoch [{epoch+1}/{n_epochs}] | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

    return train_loader, val_loader


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Loss functions
    'focal_loss',
    'combined_loss',
    'dual_decoder_loss',
    'masked_coordinate_loss',
    'chamfer_distance_loss',
    'compute_dual_decoder_loss',
    'compute_dual_decoder_chamfer_loss',
    'poisson_loss',
    'weighted_poisson_loss',

    # Target creation
    'create_density_map',
    'create_density_and_time_map',
    'density_to_coordinates',
    'density_time_to_coordinates',

    # Evaluation metrics
    'compute_iou',
    'compute_spatial_error',
    'compute_temporal_error',

    # Training functions from task4.ipynb (CNNLightningPredictor)
    'train_cnn_lightning_predictor',

    # Training functions from task4_density_probabilies.ipynb (DensityMapPredictor)
    'train_density_map_predictor',

    # Training functions from task4_transformer.ipynb (DualCNNLightningPredictor)
    'train_dual_cnn_lightning_predictor',

    # Training functions from task4-time_pred.ipynb (TimeLightningModel)
    'train_time_lightning_epoch',
    'validate_time_lightning',
    'train_time_lightning_model',

    # Training functions from task4_probability_time.ipynb (DensityTimePredictor/UNet)
    'train_density_time_predictor',

    # Training functions from task4-density_time_pred.ipynb (LightningTimePredictor)
    'train_lightning_time_epoch',
    'validate_lightning_time',
    'train_lightning_time_predictor',
]
