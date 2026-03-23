"""
Task 4 Evaluation Functions for Lightning Prediction.

This module contains evaluation functions extracted from the task4 Jupyter notebooks.
Each function is an exact copy to ensure consistency between notebook experimentation
and production code.

Usage:
    from task4.eval import evaluate_density_model, visualize_density_prediction
    from task4.model import DensityMapPredictor

    model = DensityMapPredictor()
    evaluate_density_model(model, test_loader, device)

Evaluation Functions:
    From task4.ipynb (CNNLightningPredictor):
        - evaluate_cnn_predictor: Compute time and spatial errors
        - visualize_cnn_predictions: Visualize predicted vs actual events
        - visualize_cnn_predictions_on_vil: Overlay predictions on VIL images
        - visualize_cnn_feature_maps: Visualize CNN feature maps

    From task4_density_probabilies.ipynb (DensityMapPredictor):
        - visualize_density_prediction: Visualize predicted vs actual density maps
        - visualize_predictions_on_vil: Overlay predictions on VIL images
        - evaluate_density_model: Compute IoU and focal loss metrics
        - compare_predictions_full_timespan: Compare spatial and temporal distributions

    From task4_transformer.ipynb (DualCNNLightningPredictor):
        - evaluate_dual_cnn_model: Evaluate time, spatial, and count errors

    From task4-time_pred.ipynb (TimeLightningModel):
        - plot_time_predictions_poisson: Plot predicted vs actual counts (Poisson model)
        - plot_val_examples_time: Plot validation examples for time prediction

    From task4_probability_time.ipynb (DensityTimePredictor/UNet):
        - visualize_density_time_prediction: Visualize density and time maps
        - evaluate_chamfer: Evaluate using Chamfer distance
        - compare_predictions_density_time: Compare spatial/temporal distributions

    From task4-density_time_pred.ipynb (LightningTimePredictor):
        - plot_time_predictions_mse: Plot predicted vs actual counts (MSE model)

Metric Functions:
    - compute_iou: Compute Intersection over Union
    - compute_spatial_error: Compute mean spatial error
    - compute_temporal_error: Compute mean temporal error
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import point_cloud_utils as pcu
from scipy.ndimage import maximum_filter


# =============================================================================
# Metric Functions
# =============================================================================

def compute_iou(pred, target, threshold=0.5):
    """
    Compute Intersection over Union between predicted and target density maps.

    Args:
        pred: Predicted density map (numpy array)
        target: Ground truth density map (numpy array)
        threshold: Threshold for binarization

    Returns:
        IoU score (float)
    """
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)

    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_spatial_error(pred_coords, gt_coords):
    """
    Compute mean spatial error between predicted and ground truth coordinates.

    Args:
        pred_coords: (N, 3) predicted [t, x, y]
        gt_coords: (M, 3) ground truth [t, x, y]

    Returns:
        Mean spatial error in pixels
    """
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return float('inf')

    n_compare = min(len(pred_coords), len(gt_coords))
    x_error = pred_coords[:n_compare, 1] - gt_coords[:n_compare, 1]
    y_error = pred_coords[:n_compare, 2] - gt_coords[:n_compare, 2]
    spatial_error = np.sqrt(x_error**2 + y_error**2)

    return np.mean(spatial_error)


def compute_temporal_error(pred_coords, gt_coords):
    """
    Compute mean temporal error between predicted and ground truth coordinates.

    Args:
        pred_coords: (N, 3) predicted [t, x, y] where t is in seconds
        gt_coords: (M, 3) ground truth [t, x, y]

    Returns:
        Mean temporal error in seconds
    """
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return float('inf')

    n_compare = min(len(pred_coords), len(gt_coords))
    time_error = np.abs(pred_coords[:n_compare, 0] - gt_coords[:n_compare, 0])

    return np.mean(time_error)


# =============================================================================
# From task4.ipynb - Evaluation for CNNLightningPredictor
# =============================================================================

def evaluate_cnn_predictor(model, loader, device, loader_name="Test Set"):
    """
    Evaluate CNNLightningPredictor model computing time and spatial errors.
    From: notebooks-training/task4.ipynb

    Args:
        model: CNNLightningPredictor model
        loader: DataLoader yielding (images, targets, event_counts, ids)
        device: torch device
        loader_name: Name for display purposes

    Returns:
        mean_time_error, mean_spatial_error
    """
    # Denormalization constants
    TIME_SCALE = 10800.0  # seconds
    SPATIAL_SCALE = 192.0  # pixels

    model.eval()
    all_time_errors = []
    all_spatial_errors = []

    with torch.no_grad():
        for images, targets, event_counts, ids in loader:
            images = images.to(device)
            predictions = model(images)

            for i in range(len(ids)):
                actual_count = min(event_counts[i].item(), 50)
                if actual_count > 0:
                    pred = predictions[i].cpu().numpy()
                    actual = targets[i].cpu().numpy()

                    # Denormalize predictions and targets
                    pred_time = pred[:actual_count, 0] * TIME_SCALE
                    pred_x = pred[:actual_count, 1] * SPATIAL_SCALE
                    pred_y = pred[:actual_count, 2] * SPATIAL_SCALE

                    actual_time = actual[:actual_count, 0] * TIME_SCALE
                    actual_x = actual[:actual_count, 1] * SPATIAL_SCALE
                    actual_y = actual[:actual_count, 2] * SPATIAL_SCALE

                    # Time errors (in seconds)
                    time_error = np.abs(pred_time - actual_time)
                    all_time_errors.extend(time_error)

                    # Spatial errors (Euclidean distance in pixels)
                    x_error = pred_x - actual_x
                    y_error = pred_y - actual_y
                    spatial_error = np.sqrt(x_error**2 + y_error**2)
                    all_spatial_errors.extend(spatial_error)

    mean_time_error = np.mean(all_time_errors)
    mean_spatial_error = np.mean(all_spatial_errors)

    print(f"{loader_name}:")
    print(f"  Mean Time Error: {mean_time_error:.2f} seconds ({mean_time_error/60:.2f} minutes)")
    print(f"  Mean Spatial Error: {mean_spatial_error:.2f} pixels")
    print(f"  Total predictions evaluated: {len(all_time_errors)}")

    return mean_time_error, mean_spatial_error


def visualize_cnn_predictions(model, loader, device, max_samples=None):
    """
    Visualize predicted vs actual lightning events for CNNLightningPredictor.
    From: notebooks-training/task4.ipynb

    Args:
        model: CNNLightningPredictor model
        loader: DataLoader yielding (images, targets, event_counts, ids)
        device: torch device
        max_samples: Maximum number of samples to visualize (None for all)

    Returns:
        List of dicts with 'id', 'time_error', 'spatial_error' for each sample
    """
    # Denormalization constants
    TIME_SCALE = 10800.0  # seconds
    SPATIAL_SCALE = 192.0  # pixels

    model.eval()
    results = []
    sample_count = 0

    with torch.no_grad():
        for images, targets, event_counts, ids in loader:
            images = images.to(device)
            predictions = model(images)

            # Plot for each sample in the batch
            for i in range(len(ids)):
                if max_samples is not None and sample_count >= max_samples:
                    return results

                actual_count = min(event_counts[i].item(), 50)  # Cap at max_events

                # Denormalize predictions and targets
                pred = predictions[i].cpu().numpy()
                actual = targets[i].cpu().numpy()

                pred_denorm = pred.copy()
                pred_denorm[:, 0] *= TIME_SCALE      # time back to seconds
                pred_denorm[:, 1] *= SPATIAL_SCALE   # x back to pixels
                pred_denorm[:, 2] *= SPATIAL_SCALE   # y back to pixels

                actual_denorm = actual.copy()
                actual_denorm[:, 0] *= TIME_SCALE
                actual_denorm[:, 1] *= SPATIAL_SCALE
                actual_denorm[:, 2] *= SPATIAL_SCALE

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'Test Event: {ids[i]} (Actual events: {event_counts[i].item()})', fontsize=14)

                # Plot 1: Spatial comparison (x, y locations)
                axes[0].scatter(actual_denorm[:actual_count, 1], actual_denorm[:actual_count, 2],
                               alpha=0.5, label='Actual', c='blue', s=20)
                axes[0].scatter(pred_denorm[:actual_count, 1], pred_denorm[:actual_count, 2],
                               alpha=0.5, label='Predicted', c='red', s=20, marker='x')
                axes[0].set_xlabel('X coordinate')
                axes[0].set_ylabel('Y coordinate')
                axes[0].set_title('Spatial: Predicted vs Actual Locations')
                axes[0].legend()
                axes[0].set_xlim(0, 192)
                axes[0].set_ylim(192, 0)  # Flip y-axis to match image coordinates
                axes[0].grid(True, alpha=0.3)

                # Plot 2: Temporal comparison (time values)
                axes[1].scatter(range(actual_count), actual_denorm[:actual_count, 0] / 60,
                               alpha=0.7, label='Actual', c='blue')
                axes[1].scatter(range(actual_count), pred_denorm[:actual_count, 0] / 60,
                               alpha=0.7, label='Predicted', c='red', marker='x')
                axes[1].set_xlabel('Event index')
                axes[1].set_ylabel('Time (minutes)')
                axes[1].set_title('Temporal: Predicted vs Actual Times')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

                # Compute and print summary statistics
                if actual_count > 0:
                    time_error = pred_denorm[:actual_count, 0] - actual_denorm[:actual_count, 0]
                    x_error = pred_denorm[:actual_count, 1] - actual_denorm[:actual_count, 1]
                    y_error = pred_denorm[:actual_count, 2] - actual_denorm[:actual_count, 2]
                    spatial_error = np.sqrt(x_error**2 + y_error**2)

                    mean_time_err = np.mean(np.abs(time_error))
                    mean_spatial_err = np.mean(spatial_error)

                    print(f"  Mean time error: {mean_time_err:.1f} seconds")
                    print(f"  Mean spatial error: {mean_spatial_err:.1f} pixels")
                    print("-" * 50)

                    results.append({
                        'id': ids[i],
                        'time_error': mean_time_err,
                        'spatial_error': mean_spatial_err
                    })

                sample_count += 1

    return results


def visualize_cnn_predictions_on_vil(model, loader, device, load_event_fn, max_batches=None):
    """
    Visualize predictions overlaid on VIL images in 2x5 grids.
    From: notebooks-training/task4.ipynb

    Args:
        model: CNNLightningPredictor model
        loader: DataLoader yielding (images, targets, event_counts, ids)
        device: torch device
        load_event_fn: Function to load event data by id (returns dict with 'vil' key)
        max_batches: Maximum number of 10-sample batches to display (None for all)

    Returns:
        List of sample dicts with 'id', 'pred', 'actual', 'actual_count'
    """
    from PIL import Image

    # Denormalization constants
    TIME_SCALE = 10800.0  # seconds
    SPATIAL_SCALE = 192.0  # pixels

    model.eval()

    # Collect all test samples
    all_samples = []
    with torch.no_grad():
        for images, targets, event_counts, ids in loader:
            images = images.to(device)
            predictions = model(images)

            for i in range(len(ids)):
                # Denormalize predictions and targets
                pred = predictions[i].cpu().numpy()
                actual = targets[i].cpu().numpy()

                pred_denorm = pred.copy()
                pred_denorm[:, 0] *= TIME_SCALE
                pred_denorm[:, 1] *= SPATIAL_SCALE
                pred_denorm[:, 2] *= SPATIAL_SCALE

                actual_denorm = actual.copy()
                actual_denorm[:, 0] *= TIME_SCALE
                actual_denorm[:, 1] *= SPATIAL_SCALE
                actual_denorm[:, 2] *= SPATIAL_SCALE

                all_samples.append({
                    'id': ids[i],
                    'pred': pred_denorm,
                    'actual': actual_denorm,
                    'actual_count': min(event_counts[i].item(), 50)
                })

    # Plot in 2x5 grids
    batches_plotted = 0
    for start_idx in range(0, len(all_samples), 10):
        if max_batches is not None and batches_plotted >= max_batches:
            break

        batch = all_samples[start_idx:start_idx + 10]
        n_samples = len(batch)
        n_cols = 5
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

        for idx, sample in enumerate(batch):
            ax = axes[idx]
            event = load_event_fn(sample['id'])

            actual_count = sample['actual_count']
            pred = sample['pred']
            actual = sample['actual']

            # Calculate frame from median timestamp of actual lightning events
            # Each frame = 5 minutes = 300 seconds
            if actual_count > 0:
                median_time = np.median(actual[:actual_count, 0])
                frame_idx = int(np.clip(median_time / 300, 0, 35))
            else:
                frame_idx = 17

            # Display VIL image (resize to 192x192 to match coordinate system)
            vil_frame = np.array(Image.fromarray(event['vil'][:, :, frame_idx]).resize((192, 192), Image.BILINEAR))
            ax.imshow(vil_frame, vmin=0, vmax=255, cmap='turbo')

            # Overlay actual lightning locations (blue circles)
            ax.scatter(actual[:actual_count, 1], actual[:actual_count, 2],
                      marker='o', s=15, c='blue', alpha=0.6, label='Actual')

            # Overlay predicted lightning locations (white x markers)
            ax.scatter(pred[:actual_count, 1], pred[:actual_count, 2],
                      marker='x', s=15, c='white', linewidths=1, alpha=0.9, label='Predicted')

            ax.set_xlim(0, 192)
            ax.set_ylim(192, 0)
            ax.set_title(f'{sample["id"]} (Frame {frame_idx})', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()
        batches_plotted += 1

    return all_samples


def visualize_cnn_feature_maps(model, image, device):
    """
    Extract and visualize feature maps from each conv layer.
    From: notebooks-training/task4.ipynb

    Args:
        model: CNNLightningPredictor model
        image: Single image tensor (4, 36, 192, 192) or (B, 4, 36, 192, 192)
        device: torch device
    """
    model.eval()

    # Prepare input
    if image.dim() == 4:
        image = image.unsqueeze(0)  # Add batch dim
    image = image.to(device)

    # Reshape to (batch, channels*frames, H, W)
    batch_size = image.shape[0]
    x = image.reshape(batch_size, -1, image.shape[3], image.shape[4])

    # Extract activations after each conv block
    activations = []
    layer_names = []

    for i, layer in enumerate(model.encoder):
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            activations.append(x.detach().cpu())
            layer_names.append(f'Conv {len(activations)} ({x.shape[1]} filters, {x.shape[2]}x{x.shape[3]})')

    # Plot feature maps for each conv layer
    for act, name in zip(activations, layer_names):
        n_filters = min(16, act.shape[1])  # Show up to 16 filters
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle(f'Feature Maps: {name}')

        for idx, ax in enumerate(axes.flatten()):
            if idx < n_filters:
                ax.imshow(act[0, idx], cmap='viridis')
            ax.axis('off')

        plt.tight_layout()
        plt.show()


# =============================================================================
# From task4_density_probabilies.ipynb - Evaluation for DensityMapPredictor
# =============================================================================

def visualize_density_prediction(model, test_loader, device, num_samples=4):
    """
    Visualize predicted vs actual density maps.
    From: notebooks-training/task4_density_probabilies.ipynb

    Args:
        model: DensityMapPredictor model
        test_loader: DataLoader yielding (images, density_targets, ids)
        device: torch device
        num_samples: Number of samples to visualize
    """
    model.eval()

    with torch.no_grad():
        images, density_targets, ids = next(iter(test_loader))
        images = images.to(device)
        pred_density = model(images).cpu().numpy()
        density_targets = density_targets.numpy()

        for i in range(min(num_samples, len(ids))):
            fig, axes = plt.subplots(3, 6, figsize=(18, 9))
            fig.suptitle(f'Event: {ids[i]}', fontsize=14)

            # Show 6 evenly spaced frames
            frames_to_show = [0, 7, 14, 21, 28, 35]

            for j, frame_idx in enumerate(frames_to_show):
                # Top row: VIL input (channel 3)
                axes[0, j].imshow(images[i, 3, frame_idx].cpu().numpy(), cmap='turbo', vmin=0, vmax=1)
                axes[0, j].set_title(f'VIL Frame {frame_idx}')
                axes[0, j].axis('off')

                # Middle row: Ground truth density
                axes[1, j].imshow(density_targets[i, frame_idx], cmap='hot', vmin=0, vmax=1)
                axes[1, j].set_title(f'GT Density')
                axes[1, j].axis('off')

                # Bottom row: Predicted density
                axes[2, j].imshow(pred_density[i, frame_idx], cmap='hot', vmin=0, vmax=1)
                axes[2, j].set_title(f'Pred Density')
                axes[2, j].axis('off')

            plt.tight_layout()
            plt.show()


def visualize_predictions_on_vil(model, test_loader, device, load_event_fn,
                                  density_to_coords_fn, threshold=0.3):
    """
    Visualize extracted predictions overlaid on VIL images.
    From: notebooks-training/task4_density_probabilies.ipynb

    Args:
        model: DensityMapPredictor model
        test_loader: DataLoader yielding (images, density_targets, ids)
        device: torch device
        load_event_fn: Function to load event data by id
        density_to_coords_fn: Function to convert density map to coordinates
        threshold: Detection threshold
    """
    from PIL import Image

    model.eval()

    # Collect all test samples
    all_samples = []
    with torch.no_grad():
        for images, density_targets, ids in test_loader:
            images_device = images.to(device)
            pred_density = model(images_device).cpu().numpy()

            for i in range(len(ids)):
                pred_coords = density_to_coords_fn(pred_density[i], threshold=threshold)
                all_samples.append({
                    'id': ids[i],
                    'pred_coords': pred_coords,
                    'pred_density': pred_density[i],
                    'gt_density': density_targets[i].numpy()
                })

    # Plot in 2x5 grids
    for start_idx in range(0, min(10, len(all_samples)), 10):
        batch = all_samples[start_idx:start_idx + 10]
        n_samples = len(batch)
        n_cols = 5
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

        for idx, sample in enumerate(batch):
            ax = axes[idx]
            event = load_event_fn(sample['id'])
            pred_coords = sample['pred_coords']

            # Use middle frame
            frame_idx = 17

            # Display VIL image (resize to 192x192 to match coordinate system)
            vil_frame = np.array(Image.fromarray(event['vil'][:, :, frame_idx]).resize((192, 192), Image.BILINEAR))
            ax.imshow(vil_frame, vmin=0, vmax=255, cmap='turbo')

            # Overlay actual lightning locations (blue circles)
            actual_x = event['lght'][:, 3] / 2.0
            actual_y = event['lght'][:, 4] / 2.0
            ax.scatter(actual_x, actual_y, marker='o', s=10, c='blue', alpha=0.3, label='Actual')

            # Overlay predicted lightning locations (white x markers)
            if len(pred_coords) > 0:
                ax.scatter(pred_coords[:, 1], pred_coords[:, 2],
                          marker='x', s=30, c='white', linewidths=1.5, alpha=0.9, label='Predicted')

            ax.set_xlim(0, 192)
            ax.set_ylim(192, 0)
            ax.set_title(f'{sample["id"]} ({len(pred_coords)} pred)', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()


def evaluate_density_model(model, test_loader, device, focal_loss_fn, threshold=0.5):
    """
    Compute evaluation metrics on test set for density map models.
    From: notebooks-training/task4_density_probabilies.ipynb

    Args:
        model: DensityMapPredictor model
        test_loader: DataLoader yielding (images, density_targets, ids)
        device: torch device
        focal_loss_fn: Focal loss function
        threshold: Threshold for IoU computation

    Returns:
        avg_loss, avg_iou
    """
    model.eval()

    all_ious = []
    total_loss = 0.0

    with torch.no_grad():
        for images, density_targets, ids in test_loader:
            images = images.to(device)
            density_targets_device = density_targets.to(device)

            pred_density = model(images)
            loss = focal_loss_fn(pred_density, density_targets_device)
            total_loss += loss.item()

            pred_np = pred_density.cpu().numpy()
            target_np = density_targets.numpy()

            for i in range(len(ids)):
                iou = compute_iou(pred_np[i], target_np[i], threshold)
                all_ious.append(iou)

    avg_loss = total_loss / len(test_loader)
    avg_iou = np.mean(all_ious)

    print(f"Test Results:")
    print(f"  Average Focal Loss: {avg_loss:.6f}")
    print(f"  Average IoU (threshold={threshold}): {avg_iou:.4f}")
    print(f"  Number of samples: {len(all_ious)}")

    return avg_loss, avg_iou


def compare_predictions_full_timespan(model, test_loader, device, load_event_fn,
                                       density_to_coords_fn, num_samples_to_plot=3,
                                       threshold=0.3, min_distance=5):
    """
    Visualizes and compares predicted vs actual lightning events spatially and temporally
    across the full timespan for a selection of test samples.
    From: notebooks-training/task4_density_probabilies.ipynb

    Args:
        model: DensityMapPredictor model
        test_loader: DataLoader yielding (images, density_targets, ids)
        device: torch device
        load_event_fn: Function to load event data by id
        density_to_coords_fn: Function to convert density map to coordinates
        num_samples_to_plot: Number of samples to plot
        threshold: Detection threshold
        min_distance: Minimum distance between peaks
    """
    model.eval()
    samples_plotted = 0

    with torch.no_grad():
        for images, density_targets, ids in test_loader:
            if samples_plotted >= num_samples_to_plot:
                break

            images_device = images.to(device)
            pred_density = model(images_device).cpu().numpy()

            for i in range(len(ids)):
                if samples_plotted >= num_samples_to_plot:
                    break

                event_id = ids[i]

                # Get predicted coordinates from density map
                current_pred_density = pred_density[i]  # (T, H, W)
                pred_coords = density_to_coords_fn(current_pred_density, threshold=threshold, min_distance=min_distance)

                # Load actual event to get true lightning coordinates
                event_data = load_event_fn(event_id)
                actual_lght = event_data['lght']

                # Scale actual coordinates to 192x192 resolution
                actual_coords = np.stack([
                    actual_lght[:, 0],           # time in seconds
                    actual_lght[:, 3] / 2.0,     # x scaled to 192
                    actual_lght[:, 4] / 2.0      # y scaled to 192
                ], axis=1).astype(np.float32)

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.suptitle(f'Event: {event_id} - Predicted vs Actual Lightning', fontsize=16)

                # --- Spatial Comparison (Scatter Plot of X, Y coordinates) ---
                if len(actual_coords) > 0:
                    axes[0].scatter(actual_coords[:, 1], actual_coords[:, 2],
                                    alpha=0.5, label=f'Actual ({len(actual_coords)})', c='blue', s=20)
                if len(pred_coords) > 0:
                    axes[0].scatter(pred_coords[:, 1], pred_coords[:, 2],
                                    alpha=0.5, label=f'Predicted ({len(pred_coords)})', c='red', s=20, marker='x')

                axes[0].set_xlabel('X coordinate')
                axes[0].set_ylabel('Y coordinate')
                axes[0].set_title('Spatial Comparison')
                axes[0].set_xlim(0, 192)
                axes[0].set_ylim(192, 0)  # Flip y-axis to match image coordinates convention
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # --- Temporal Comparison (Histograms of Time) ---
                bins = np.arange(0, 36 * 5 + 1, 5)  # 0 to 180 minutes, 5-min intervals

                if len(actual_coords) > 0:
                    axes[1].hist(actual_coords[:, 0] / 60.0, bins=bins, alpha=0.6, label='Actual', color='blue', edgecolor='black')
                if len(pred_coords) > 0:
                    axes[1].hist(pred_coords[:, 0] / 60.0, bins=bins, alpha=0.6, label='Predicted', color='red', edgecolor='black')

                axes[1].set_xlabel('Time (minutes)')
                axes[1].set_ylabel('Number of Lightning Strikes')
                axes[1].set_title('Temporal Comparison (5-min bins)')
                axes[1].set_xlim(0, 180)  # 36 frames * 5 minutes/frame
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
                samples_plotted += 1


# =============================================================================
# From task4_transformer.ipynb - Evaluation for DualCNNLightningPredictor
# =============================================================================

def evaluate_dual_cnn_model(model, loader, device, loader_name="Test Set"):
    """
    Evaluate dual-CNN model on a data loader.
    From: notebooks-training/task4_transformer.ipynb

    Args:
        model: DualCNNLightningPredictor model with .generate() method
        loader: DataLoader yielding (images, targets, mask, strike_counts, ids)
        device: torch device
        loader_name: Name for display purposes

    Returns:
        all_time_errors, all_spatial_errors, all_count_errors
    """
    # Denormalization constants
    TIME_SCALE = 10800.0  # seconds
    SPATIAL_SCALE = 192.0  # pixels

    model.eval()
    all_time_errors = []
    all_spatial_errors = []
    all_count_errors = []

    with torch.no_grad():
        for images, targets, mask, strike_counts, ids in loader:
            images = images.to(device)

            # Generate predictions
            predictions = model.generate(images, max_strikes=50000, stop_threshold=0.5)

            for i in range(len(ids)):
                actual_count = strike_counts[i].item()
                pred_strikes = predictions[i].numpy()  # (N_pred, 3)
                actual_strikes = targets[i, :actual_count].numpy()  # (N_actual, 3)

                pred_count = len(pred_strikes)
                all_count_errors.append(abs(pred_count - actual_count))

                if pred_count > 0 and actual_count > 0:
                    # Compare first min(pred, actual) strikes
                    n_compare = min(pred_count, actual_count)

                    # Denormalize
                    pred_t = pred_strikes[:n_compare, 0] * TIME_SCALE
                    pred_x = pred_strikes[:n_compare, 1] * SPATIAL_SCALE
                    pred_y = pred_strikes[:n_compare, 2] * SPATIAL_SCALE

                    actual_t = actual_strikes[:n_compare, 0] * TIME_SCALE
                    actual_x = actual_strikes[:n_compare, 1] * SPATIAL_SCALE
                    actual_y = actual_strikes[:n_compare, 2] * SPATIAL_SCALE

                    # Errors
                    time_error = np.abs(pred_t - actual_t)
                    spatial_error = np.sqrt((pred_x - actual_x)**2 + (pred_y - actual_y)**2)

                    all_time_errors.extend(time_error)
                    all_spatial_errors.extend(spatial_error)

    print(f"\n{loader_name} Results (Dual-CNN):")
    print(f"  Mean Time Error: {np.mean(all_time_errors):.1f} seconds ({np.mean(all_time_errors)/60:.1f} minutes)")
    print(f"  Mean Spatial Error: {np.mean(all_spatial_errors):.1f} pixels")
    print(f"  Mean Count Error: {np.mean(all_count_errors):.1f} strikes")

    return all_time_errors, all_spatial_errors, all_count_errors


# =============================================================================
# From task4-time_pred.ipynb - Evaluation for TimeLightningModel
# =============================================================================

def plot_time_predictions_poisson(model, loader, device, n_examples=3):
    """
    Plot predicted vs actual lightning counts for Poisson-based models.
    From: notebooks-training/task4-time_pred.ipynb

    Args:
        model: TimeLightningModel that outputs log(lambda)
        loader: DataLoader yielding (images, counts, event_ids)
        device: torch device
        n_examples: Number of examples to plot
    """
    model.eval()

    with torch.no_grad():
        images, counts, event_ids = next(iter(loader))
        images = images.to(device)
        counts = counts.to(device)

        outputs = model(images)
        preds = torch.exp(outputs)  # expected counts (lambda)

        n_examples = min(n_examples, images.size(0))
        time_axis = np.arange(counts.size(1)) * 5  # minutes (36 * 5min)

        for i in range(n_examples):
            plt.figure(figsize=(10, 4))

            plt.step(
                time_axis,
                counts[i].cpu().numpy(),
                where="post",
                label="Ground truth",
                linewidth=2,
            )
            plt.plot(
                time_axis,
                preds[i].cpu().numpy(),
                label="Prediction (expected count)",
                linewidth=2,
            )

            plt.xlabel("Time (minutes)")
            plt.ylabel("Lightning count")
            plt.title(f"Event {event_ids[i]}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()


def plot_val_examples_time(model, loader, device, n_batches=1, n_per_batch=2):
    """
    Plot validation examples for time prediction models.
    From: notebooks-training/task4-time_pred.ipynb

    Args:
        model: Time prediction model
        loader: DataLoader yielding (images, targets)
        device: torch device
        n_batches: Number of batches to process
        n_per_batch: Number of samples per batch to plot
    """
    model.eval()
    count = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            targets = targets.numpy()

            for i in range(min(n_per_batch, len(images))):
                plt.figure(figsize=(10, 3))
                plt.plot(targets[i], label='True Count', color='black', alpha=0.5, drawstyle='steps-mid')
                plt.plot(outputs[i], label='Predicted Count', color='red', lw=2)
                plt.title(f"Example Storm Event Timeline")
                plt.xlabel("Frame (5-min intervals)")
                plt.ylabel("Lightning Strikes")
                plt.legend()
                plt.show()

            count += 1
            if count >= n_batches:
                break


# =============================================================================
# From task4_probability_time.ipynb - Evaluation for DensityTimePredictor/UNet
# =============================================================================

def visualize_density_time_prediction(model, test_loader, device, num_samples=3):
    """
    Visualize predicted vs actual per-frame density and time maps.
    From: notebooks-training/task4_probability_time.ipynb

    Args:
        model: DensityTimePredictor or DensityTimeUNet model
        test_loader: DataLoader yielding (images, targets, ids)
        device: torch device
        num_samples: Number of samples to visualize
    """
    model.eval()

    selected_frames = [0, 8, 17, 26, 35]

    with torch.no_grad():
        images, targets, ids = next(iter(test_loader))
        images = images.to(device)
        pred = model(images).cpu().numpy()
        targets = targets.numpy()

        for i in range(min(num_samples, len(ids))):
            n_frames = len(selected_frames)
            fig, axes = plt.subplots(4, n_frames, figsize=(4 * n_frames, 14))
            fig.suptitle(f'Event: {ids[i]}', fontsize=14)

            for col, f in enumerate(selected_frames):
                # Row 0: VIL input
                axes[0, col].imshow(images[i, 3, f].cpu().numpy(), cmap='turbo', vmin=0, vmax=1)
                axes[0, col].set_title(f'VIL Frame {f}\n({f*5}-{(f+1)*5} min)')
                axes[0, col].axis('off')

                # Row 1: GT Density
                axes[1, col].imshow(targets[i, 0, f], cmap='hot', vmin=0, vmax=1)
                axes[1, col].set_title(f'GT Density')
                axes[1, col].axis('off')

                # Row 2: Pred Density
                axes[2, col].imshow(pred[i, 0, f], cmap='hot', vmin=0, vmax=1)
                axes[2, col].set_title(f'Pred Density')
                axes[2, col].axis('off')

                # Row 3: Pred Time
                pred_time_display = np.ma.masked_where(pred[i, 0, f] < 0.01, pred[i, 1, f])
                axes[3, col].imshow(pred_time_display, cmap='viridis', vmin=0, vmax=1)
                axes[3, col].set_title(f'Pred Time [0-1]')
                axes[3, col].axis('off')

            axes[0, 0].set_ylabel('VIL', fontsize=11)
            axes[1, 0].set_ylabel('GT Density', fontsize=11)
            axes[2, 0].set_ylabel('Pred Density', fontsize=11)
            axes[3, 0].set_ylabel('Pred Time', fontsize=11)

            plt.tight_layout()
            plt.show()


def evaluate_chamfer(model, test_loader, device, load_event_fn, density_to_coords_fn,
                      threshold=0.2):
    """
    Evaluate model using Chamfer distance on extracted coordinates.
    From: notebooks-training/task4_probability_time.ipynb

    Note: Requires point_cloud_utils (pcu) package.

    Args:
        model: Density prediction model
        test_loader: DataLoader yielding (images, targets, ids)
        device: torch device
        load_event_fn: Function to load event data by id
        density_to_coords_fn: Function to convert density map to coordinates
        threshold: Detection threshold

    Returns:
        mean_spatial_cd, mean_time_cd, mean_full_cd
    """
    model.eval()
    spatial_cds = []
    time_cds = []
    full_cds = []

    with torch.no_grad():
        for images, targets, ids in test_loader:
            images_device = images.to(device)
            pred = model(images_device).cpu().numpy()

            for i in range(len(ids)):
                # Extract predicted coordinates from density+time map
                pred_coords = density_to_coords_fn(pred[i], threshold=threshold)

                # Get actual coordinates
                event = load_event_fn(ids[i])
                actual_coords = np.stack([
                    event['lght'][:, 0],
                    event['lght'][:, 3] / 2.0,
                    event['lght'][:, 4] / 2.0
                ], axis=1).astype(np.float32)

                if len(pred_coords) == 0 or len(actual_coords) == 0:
                    continue

                # pcu.chamfer_distance expects (N, 3) float32 arrays
                # Pad 2D arrays to 3D with zeros for spatial-only / time-only

                # Spatial Chamfer (x, y only) — pad with zero z-column
                pred_spatial = np.column_stack([pred_coords[:, 1:], np.zeros(len(pred_coords))]).astype(np.float32)
                actual_spatial = np.column_stack([actual_coords[:, 1:], np.zeros(len(actual_coords))]).astype(np.float32)
                cd_s = pcu.chamfer_distance(pred_spatial, actual_spatial)
                spatial_cds.append(cd_s)

                # Temporal Chamfer (time only, in minutes) — pad with zero y,z columns
                pred_time = np.column_stack([pred_coords[:, 0:1] / 60.0, np.zeros((len(pred_coords), 2))]).astype(np.float32)
                actual_time = np.column_stack([actual_coords[:, 0:1] / 60.0, np.zeros((len(actual_coords), 2))]).astype(np.float32)
                cd_t = pcu.chamfer_distance(pred_time, actual_time)
                time_cds.append(cd_t)

                # Full Chamfer (t_minutes, x, y)
                pred_full = pred_coords.copy().astype(np.float32)
                actual_full = actual_coords.copy().astype(np.float32)
                cd_f = pcu.chamfer_distance(pred_full, actual_full)
                full_cds.append(cd_f)

    print("Chamfer Distance Results:")
    print(f"  Spatial CD (pixels²):   {np.mean(spatial_cds):.2f} ± {np.std(spatial_cds):.2f}")
    print(f"  Temporal CD (minutes²): {np.mean(time_cds):.2f} ± {np.std(time_cds):.2f}")
    print(f"  Full CD (t,x,y):       {np.mean(full_cds):.2f} ± {np.std(full_cds):.2f}")
    print(f"  Samples evaluated: {len(spatial_cds)}")

    return np.mean(spatial_cds), np.mean(time_cds), np.mean(full_cds)


def compare_predictions_density_time(model, test_loader, device, load_event_fn,
                                      density_to_coords_fn, num_samples=4, threshold=0.3):
    """
    Compare predicted vs actual lightning events spatially and temporally.
    From: notebooks-training/task4_probability_time.ipynb

    Args:
        model: Density+time prediction model
        test_loader: DataLoader yielding (images, targets, ids)
        device: torch device
        load_event_fn: Function to load event data by id
        density_to_coords_fn: Function to convert density map to coordinates
        num_samples: Number of samples to plot
        threshold: Detection threshold
    """
    model.eval()
    samples_plotted = 0

    with torch.no_grad():
        for images, targets, ids in test_loader:
            if samples_plotted >= num_samples:
                break

            images_device = images.to(device)
            pred = model(images_device).cpu().numpy()

            for i in range(len(ids)):
                if samples_plotted >= num_samples:
                    break

                event_id = ids[i]
                pred_coords = density_to_coords_fn(pred[i], threshold=threshold)

                event_data = load_event_fn(event_id)
                actual_coords = np.stack([
                    event_data['lght'][:, 0],
                    event_data['lght'][:, 3] / 2.0,
                    event_data['lght'][:, 4] / 2.0
                ], axis=1).astype(np.float32)

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.suptitle(f'Event: {event_id} — Predicted vs Actual', fontsize=16)

                # Spatial comparison
                if len(actual_coords) > 0:
                    axes[0].scatter(actual_coords[:, 1], actual_coords[:, 2],
                                    alpha=0.5, label=f'Actual ({len(actual_coords)})', c='blue', s=20)
                if len(pred_coords) > 0:
                    axes[0].scatter(pred_coords[:, 1], pred_coords[:, 2],
                                    alpha=0.5, label=f'Predicted ({len(pred_coords)})', c='red', s=20, marker='x')

                axes[0].set_xlabel('X coordinate')
                axes[0].set_ylabel('Y coordinate')
                axes[0].set_title('Spatial Comparison')
                axes[0].set_xlim(0, 192)
                axes[0].set_ylim(192, 0)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # Temporal comparison
                bins = np.arange(0, 36 * 5 + 1, 5)

                if len(actual_coords) > 0:
                    axes[1].hist(actual_coords[:, 0] / 60.0, bins=bins, alpha=0.6,
                                 label='Actual', color='blue', edgecolor='black')
                if len(pred_coords) > 0:
                    axes[1].hist(pred_coords[:, 0] / 60.0, bins=bins, alpha=0.6,
                                 label='Predicted', color='red', edgecolor='black')

                axes[1].set_xlabel('Time (minutes)')
                axes[1].set_ylabel('Number of Lightning Strikes')
                axes[1].set_title('Temporal Comparison (5-min bins)')
                axes[1].set_xlim(0, 180)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()
                samples_plotted += 1


# =============================================================================
# From task4-density_time_pred.ipynb - Evaluation for LightningTimePredictor
# =============================================================================

def plot_time_predictions_mse(model, loader, device, n_examples=3, num_bins=36):
    """
    Plots predicted vs ground truth lightning counts over time for MSE-based models.
    From: notebooks-training/task4-density_time_pred.ipynb

    Args:
        model: LightningTimePredictor model
        loader: DataLoader returning (positions, masks, time_counts, event_ids)
        device: torch.device
        n_examples: Number of events to plot
        num_bins: Number of time bins
    """
    model.eval()

    with torch.no_grad():
        # Get a single batch
        positions, masks, counts, event_ids = next(iter(loader))
        positions = positions.to(device)
        masks = masks.to(device)
        counts = counts.to(device)

        # Model predictions
        outputs = model(positions, masks)  # (B, num_bins)
        preds = outputs  # Already non-negative counts due to ReLU

        n_examples = min(n_examples, positions.size(0))
        time_axis = np.arange(counts.size(1)) * (3*60/num_bins)  # minutes

        for i in range(n_examples):
            plt.figure(figsize=(10, 4))
            # Ground truth
            plt.step(
                time_axis,
                counts[i].cpu().numpy(),
                where="post",
                label="Ground truth",
                linewidth=2,
                color="blue"
            )
            plt.step(
                time_axis,
                preds[i].cpu().numpy(),
                where="post",
                label="Prediction",
                linewidth=2,
                color="red"
            )

            plt.xlabel("Time (minutes)")
            plt.ylabel("Lightning count")
            plt.title(f"Event {event_ids[i]}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()


def visualize_nth_event_distribution(data_list, n=50):
    """
    Visualize the distribution of the Nth lightning event time across samples.
    From: notebooks-training/task4-datavis-baseline.ipynb

    Args:
        data_list: List of (images, targets, event_id) tuples where targets is (N, 3)
                   with columns [time, x, y]
        n: Which event to analyze (default 50, meaning the 50th lightning strike)

    Returns:
        times: List of times for the Nth event
        event_ids: List of event IDs that have at least N strikes
    """
    times_of_nth_event = []
    event_ids_with_n = []

    for images, targets, event_id in data_list:
        # targets shape: (N, 3) where N is number of events, columns are [time, x, y]
        if targets.shape[0] >= n:
            # Get time of nth event (0-indexed)
            time_nth = targets[n - 1, 0]  # time in seconds
            times_of_nth_event.append(time_nth)
            event_ids_with_n.append(event_id)

    print(f"Events with at least {n} lightning strikes: {len(times_of_nth_event)} / {len(data_list)}")

    if times_of_nth_event:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(np.array(times_of_nth_event) / 60, bins=20, edgecolor='black')
        plt.xlabel(f'Time of {n}th Event (minutes)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {n}th Lightning Event Time')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(range(len(times_of_nth_event)), np.array(times_of_nth_event) / 60, alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel(f'Time of {n}th Event (minutes)')
        plt.title(f'{n}th Lightning Event Time per Sample')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"Mean time of {n}th event: {np.mean(times_of_nth_event) / 60:.2f} minutes")
        print(f"Min: {np.min(times_of_nth_event) / 60:.2f} min, Max: {np.max(times_of_nth_event) / 60:.2f} min")
    else:
        print(f"No events in data have {n} or more lightning strikes.")

    return times_of_nth_event, event_ids_with_n


def visualize_nth_event_distribution_all(event_ids, load_event_fn, n=50):
    """
    Visualize the distribution of the Nth lightning event time across all events.
    Loads events directly using load_event function.
    From: notebooks-training/task4-datavis-baseline.ipynb

    Args:
        event_ids: List/array of event IDs to analyze
        load_event_fn: Function to load event data by id (returns dict with 'lght' key)
        n: Which event to analyze (default 50, meaning the 50th lightning strike)

    Returns:
        times: List of times for the Nth event
        event_ids_with_n: List of event IDs that have at least N strikes
    """
    times_of_nth_event = []
    event_ids_with_n = []

    for event_id in event_ids:
        event = load_event_fn(event_id)
        lght = event['lght']  # (N, 5) - [time, lat, lon, x, y]

        if lght.shape[0] >= n:
            # Get time of nth event (0-indexed)
            time_nth = lght[n - 1, 0]  # time in seconds
            times_of_nth_event.append(time_nth)
            event_ids_with_n.append(event_id)

    print(f"Events with at least {n} lightning strikes: {len(times_of_nth_event)} / {len(event_ids)}")

    if times_of_nth_event:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(np.array(times_of_nth_event) / 60, bins=30, edgecolor='black')
        plt.xlabel(f'Time of {n}th Event (minutes)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {n}th Lightning Event Time (All {len(event_ids)} Events)')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(range(len(times_of_nth_event)), np.array(times_of_nth_event) / 60, alpha=0.5, s=10)
        plt.xlabel('Sample Index')
        plt.ylabel(f'Time of {n}th Event (minutes)')
        plt.title(f'{n}th Lightning Event Time per Sample')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"Mean time of {n}th event: {np.mean(times_of_nth_event) / 60:.2f} minutes")
        print(f"Median time of {n}th event: {np.median(times_of_nth_event) / 60:.2f} minutes")
        print(f"Min: {np.min(times_of_nth_event) / 60:.2f} min, Max: {np.max(times_of_nth_event) / 60:.2f} min")
        print(f"Std: {np.std(times_of_nth_event) / 60:.2f} minutes")
    else:
        print(f"No events have {n} or more lightning strikes.")

    return times_of_nth_event, event_ids_with_n


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Metric functions
    'compute_iou',
    'compute_spatial_error',
    'compute_temporal_error',

    # From task4.ipynb (CNNLightningPredictor)
    'evaluate_cnn_predictor',
    'visualize_cnn_predictions',
    'visualize_cnn_predictions_on_vil',
    'visualize_cnn_feature_maps',

    # From task4_density_probabilies.ipynb (DensityMapPredictor)
    'visualize_density_prediction',
    'visualize_predictions_on_vil',
    'evaluate_density_model',
    'compare_predictions_full_timespan',

    # From task4_transformer.ipynb (DualCNNLightningPredictor)
    'evaluate_dual_cnn_model',

    # From task4-time_pred.ipynb (TimeLightningModel)
    'plot_time_predictions_poisson',
    'plot_val_examples_time',

    # From task4_probability_time.ipynb (DensityTimePredictor/UNet)
    'visualize_density_time_prediction',
    'evaluate_chamfer',
    'compare_predictions_density_time',

    # From task4-density_time_pred.ipynb (LightningTimePredictor)
    'plot_time_predictions_mse',

    # From task4-datavis-baseline.ipynb (data analysis)
    'visualize_nth_event_distribution',
    'visualize_nth_event_distribution_all',
]
