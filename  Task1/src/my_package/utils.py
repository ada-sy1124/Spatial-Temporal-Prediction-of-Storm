import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib import animation
import h5py


def show_fixed_event_prediction(
    model,
    val_dataset,
    device,
    fixed_event_idx=0,
    n_frames=12,
    ckpt_path=None,
    cmap="turbo",
    vmin=0,
    vmax=1,
    figsize=(20, 14),
):
    """
    This function takes a fixed sample (fixed_event_idx) from the `val_dataset` and has the model predict and plot a 12-frame comparison of ground truth (GT) versus pre-defined (FRD).
    Args:
    model: Your PyTorch model (already built)
    val_dataset: A Dataset object (__getitem__ returns (x_in, x_out))
    device: "cuda" or "cpu"
    fixed_event_idx: The sample number you want to see (int)
    n_frames: The number of frames to predict/plot (default 12)
    ckpt_path: Optional; if provided, `load_state_dict` will be loaded before prediction
    cmap, vmin, vmax: Display parameters for `imshow`
    figsize: Image size

    Returns:
        inputs_origin, targets_origin, preds
        - inputs_origin: (T_in, 1, H, W) tensor (cpu)
        - targets_origin: (T_out, 1, H, W) tensor (cpu)
        - preds: (1, T_out, 1, H, W) tensor (device)
    """
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

    inputs_origin, targets_origin = val_dataset[fixed_event_idx]  # (T,1,H,W)

    inputs = inputs_origin.unsqueeze(0).to(device)  # (1,T,1,H,W)

    model.eval()
    with torch.no_grad():
        preds = model(inputs)

    true_seq = targets_origin.squeeze().cpu().numpy()
    pred_seq = preds[0].squeeze().detach().cpu().numpy()

    if true_seq.ndim == 2:
        true_seq = true_seq[None, ...]
    if pred_seq.ndim == 2:
        pred_seq = pred_seq[None, ...]

    T = min(n_frames, true_seq.shape[0], pred_seq.shape[0])

    fig, axes = plt.subplots(4, 6, figsize=figsize)
    plt.suptitle(
        f"Fixed Evaluation - Event IDX: {fixed_event_idx} (First {T} Frames)",
        fontsize=20,
        y=0.96
    )

    for t in range(12):
        col = t % 6

        # True rows: 0,1
        row_true = 0 if t < 6 else 1
        ax_t = axes[row_true, col]
        if t < T:
            ax_t.imshow(true_seq[t], cmap=cmap, vmin=vmin, vmax=vmax)
            ax_t.set_title(f"True T={t+1}", fontsize=10)
        ax_t.axis("off")

        # Pred rows: 2,3
        row_pred = 2 if t < 6 else 3
        ax_p = axes[row_pred, col]
        if t < T:
            ax_p.imshow(pred_seq[t], cmap=cmap, vmin=vmin, vmax=vmax)
            ax_p.set_title(f"Pred T={t+1}", fontsize=10)
        ax_p.axis("off")

    plt.figtext(
        0.5, 0.51, f" PREDICTION RESULTS (IDX {fixed_event_idx}) ",
        ha="center", va="center", fontsize=16, weight="bold",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray", "pad": 5}
    )
    plt.figtext(
        0.5, 0.92, f" GROUND TRUTH (IDX {fixed_event_idx}) ",
        ha="center", va="center", fontsize=16, weight="bold", color="green"
    )

    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    plt.show()

    return inputs_origin, targets_origin, preds


def create_prediction_gif(val_dataset, model, device,
                          filename='prediction.gif',  idx=0):
    """
    Create GIFs comparing ground truth with predicted sequence.
    Args:
        real_seq (torch.Tensor): Real image sequence of shape (T, H, W)
                                                         or (T, 1, H, W).
        pred_seq (torch.Tensor): Predicted image sequence of shape (T, H, W)
                                                         or (T, 1, H, W).
        filename (str): Output filename for the GIF.
    Returns:
        filename

    Use it like this :
    from IPython.display import Image
    gif_path = create_prediction_gif(val_dataset, model, device)
    Image(open("prediction.gif",'rb').read())

    """
    # 1. Get the data from the dataset
    inputs_origin, targets_origin = val_dataset[idx]

    # 2. Prepare for model (add batch dim: [1, 12, 1, H, W])
    inputs = inputs_origin.unsqueeze(0).to(device)

    # 3. Inference
    model.eval()  # Good practice to set eval mode
    with torch.no_grad():
        preds = model(inputs)

    # 4. Process for plotting: Remove extra dims and move to CPU
    # targets_origin is (out_len, 1, H, W) ->
    #       squeeze(1) makes it (out_len, H, W)
    real_imgs = targets_origin.squeeze(1).cpu().numpy()

    # preds is usually (1, out_len, 1, H, W) ->
    #       squeeze() makes it (out_len, H, W)
    pred_imgs = preds.squeeze().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def init():
        ax1.clear()
        ax2.clear()
        return []

    def animate(frame):
        ax1.clear()
        ax2.clear()
        # Ensure we don't index out of bounds
        ax1.imshow(real_imgs[frame], cmap='turbo')
        ax1.set_title(f'Real - Frame {frame+1}')
        ax1.axis('off')

        ax2.imshow(pred_imgs[frame], cmap='turbo')
        ax2.set_title(f'Predicted - Frame {frame+1}')
        ax2.axis('off')
        return []

    anim = animation.FuncAnimation(
                            fig, animate, init_func=init,
                            frames=len(real_imgs), interval=200, blit=False) # blit=False is safer for clear()
    anim.save(filename)
    plt.close()
    return filename


def plot_prediction_metrics_all_events(model, val_dataset, device):
    """
    Evaluates and plots the average error metrics (MAE, MSE, RMSE) across
    the entire validation dataset as a function of lead time.

    The function iterates through every sequence in the validation set,
    generates predictions, and calculates pixel-wise errors for each of
    the 12 future timesteps. It then aggregates these errors to visualize
    how prediction accuracy decreases over time (prediction decay).

    Args:
        model (torch.nn.Module): The trained predictive model (e.g., RNN,
            PredRNN, LSTM) to be evaluated.
        val_dataset (torch.utils.data.Dataset): The validation dataset where
            each item returns a tuple of (inputs, targets).
        device (torch.device): The device (CPU or CUDA) to perform
                                    inference on.

    Returns:
        None: Displays a matplotlib figure showing the average MAE, MSE,
        and RMSE trends over 60 minutes of lead time.

    Notes:
        - The function assumes a fixed output sequence length of 12 frames.
        - X-axis lead times are hardcoded for 5-minute intervals (5 to 60 min).
        - Metrics are calculated in image space (pixel-wise) before averaging.
    """

    # 1. Setup accumulators
    num_val_events = len(val_dataset)
    total_mae = np.zeros(12)
    total_mse = np.zeros(12)
    total_rmse = np.zeros(12)

    model.eval()
    print(f"Calculating average metrics for {num_val_events} events...")

    # 2. Iterate through the entire validation dataset
    with torch.no_grad():
        for idx in tqdm(range(num_val_events)):
            # Get data (T, C, H, W)
            inputs_origin, targets_origin = val_dataset[idx]

            # Prepare for model (add batch dim and send to device)
            inputs = inputs_origin.unsqueeze(0).to(device)

            # Inference
            preds = model(inputs)

            # Convert to numpy for metric calculation (12, H, W)
            true_seq = targets_origin.squeeze().cpu().numpy()
            pred_seq = preds[0].squeeze().cpu().numpy()

            # Calculate metrics for each of the 12 timesteps for THIS event
            for t in range(12):
                mae = np.mean(np.abs(true_seq[t] - pred_seq[t]))
                mse = np.mean((true_seq[t] - pred_seq[t])**2)
                rmse = np.sqrt(mse)

                total_mae[t] += mae
                total_mse[t] += mse
                total_rmse[t] += rmse

    # 3. Compute the final average across all events
    avg_mae = total_mae / num_val_events
    avg_mse = total_mse / num_val_events
    avg_rmse = total_rmse / num_val_events

    # --- Plotting Global Performance ---
    plt.figure(figsize=(10, 6))

    time_steps = np.arange(5, 65, 5)

    plt.plot(time_steps, avg_mae, marker='o', linestyle='-',
             linewidth=2, label='Avg MAE', color='#1f77b4')
    plt.plot(time_steps, avg_mse, marker='d', linestyle='-',
             linewidth=2, label='Avg MSE', color='#ff7f0e')
    plt.plot(time_steps, avg_rmse, marker='s', linestyle='--',
             linewidth=2, label='Avg RMSE', color='#2ca02c')

    plt.title(
        f'Global Prediction Decay (Average of {num_val_events} Validation Events)', fontsize=14)
    plt.xlabel('Lead Time (Minutes)', fontsize=12)
    plt.ylabel('Error Metric', fontsize=12)

    plt.xticks(time_steps)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper left', frameon=True)

    # Add a subtle note about the dataset size
    plt.figtext(0.99, 0.01, f'N={num_val_events}', horizontalalignment='right',
                fontsize=9, alpha=0.6)

    plt.tight_layout()
    plt.show()


# utils.py


@torch.no_grad()
def evaluate_persistence_baseline(
    dataloader,
    device=None,
    pixel_scale=255.0,
    verbose=True
):
    """
    Evaluate the persistence baseline on a given dataloader.
    The persistence baseline assumes that the future state will be identical
    to the last observed frame. This is a common benchmark for spatio-temporal
    prediction tasks.

    Args:
        dataloader (DataLoader): Data provider yielding (x_in, x_out).
        device (torch.device, optional): Computational device.
        pixel_scale (float): Multiplier to convert normalized MAE to pixel
                            scale.
        verbose (bool): If True, prints the results summary.

    Returns:
        dict: Dictionary containing MAE, MSE, and pixel-scaled MAE.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mae_list = []
    mse_list = []

    for x_in, x_out in tqdm(dataloader, desc="Evaluating Persistence", leave=False):
        x_in = x_in.to(device)
        x_out = x_out.to(device)

        # Last input frame shape: (B, 1, H, W)
        last_frame = x_in[:, -1]

        # Expand last frame to match x_out shape: (B, T_out, 1, H, W) via broadcasting
        last_frame_expanded = last_frame.unsqueeze(1)

        diff = last_frame_expanded - x_out

        batch_mae = torch.abs(diff).mean().item()
        batch_mse = (diff ** 2).mean().item()

        mae_list.append(batch_mae)
        mse_list.append(batch_mse)

    final_mae = float(np.mean(mae_list))
    final_mse = float(np.mean(mse_list))
    pixel_mae = final_mae * pixel_scale

    if verbose:
        print("Persistence Baseline Results:")
        print(f"  Normalized MAE : {final_mae:.6f}")
        print(f"  Normalized MSE : {final_mse:.6f}")
        print(f"  Pixel MAE      : {pixel_mae:.2f}")

    return {
        "mae": final_mae,
        "mse": final_mse,
        "pixel_mae": pixel_mae,
    }


# ==========================================================
# Streaming Statistics
# ==========================================================
class StreamingStats:
    """
    Computes mean, standard deviation, and histograms over large datasets 
    without loading the entire dataset into memory.
    """
    def __init__(self, n_bins=100, val_range=(0, 1)):
        self.n_bins = n_bins
        self.val_range = val_range

        self.total_sum = 0.0
        self.total_sq_sum = 0.0
        self.total_count = 0

        self.hist_counts = np.zeros(n_bins, dtype=np.int64)
        self.bin_edges = np.linspace(val_range[0], val_range[1], n_bins + 1)

    def update(self, batch_data):
        """Updates internal statistics with a new batch of data."""
        if isinstance(batch_data, torch.Tensor):
            data = batch_data.detach().cpu().numpy().ravel()
        else:
            data = batch_data.ravel()

        self.total_sum += np.sum(data, dtype=np.float64)
        self.total_sq_sum += np.sum(data ** 2, dtype=np.float64)
        self.total_count += data.size

        counts, _ = np.histogram(data, bins=self.bin_edges)
        self.hist_counts += counts

    def compute(self):
        """Returns the computed mean, std, bin centers, and density."""
        if self.total_count == 0:
            return 0.0, 0.0, None, None

        mean = self.total_sum / self.total_count
        var = (self.total_sq_sum / self.total_count) - mean ** 2
        std = np.sqrt(max(var, 0.0))

        bin_width = self.bin_edges[1] - self.bin_edges[0]
        density = self.hist_counts / (self.total_count * bin_width)
        bin_centers = 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])

        return mean, std, bin_centers, density


# ==========================================================
# ONE-LINE API: plot pixel distribution
# ==========================================================
@torch.no_grad()
def plot_pixel_distribution(
    train_loader,
    val_loader=None,
    n_bins=100,
    val_range=(0, 1),
    log_scale=True
):
    """
    Scans through training and validation loaders to plot the distribution 
    of pixel values (VIL) and calculate global statistics.
    """
    train_stats = StreamingStats(n_bins=n_bins, val_range=val_range)
    val_stats = StreamingStats(n_bins=n_bins, val_range=val_range) if val_loader is not None else None

    # -------- Train --------
    for x_in, x_out in tqdm(train_loader, desc="Scanning Train"):
        train_stats.update(x_in)
        train_stats.update(x_out)

    train_mean, train_std, train_x, train_y = train_stats.compute()

    # -------- Val --------
    if val_loader is not None:
        for x_in, x_out in tqdm(val_loader, desc="Scanning Val"):
            val_stats.update(x_in)
            val_stats.update(x_out)

        val_mean, val_std, val_x, val_y = val_stats.compute()

    # -------- Print summary --------
    print(f"{'Dataset':<10} {'Mean':<10} {'Std':<10}")
    print(f"{'Train':<10} {train_mean:.6f} {train_std:.6f}")
    if val_loader is not None:
        print(f"{'Val':<10} {val_mean:.6f} {val_std:.6f}")

    # -------- Plot --------
    plt.figure(figsize=(10, 5))

    plt.plot(
        train_x, train_y,
        label=f"Train (μ={train_mean:.3f}, σ={train_std:.3f})",
        color="blue", alpha=0.8
    )
    plt.fill_between(train_x, train_y, color="blue", alpha=0.1)

    if val_loader is not None:
        plt.plot(
            val_x, val_y,
            label=f"Val (μ={val_mean:.3f}, σ={val_std:.3f})",
            color="orange", linestyle="--", alpha=0.8
        )

    if log_scale:
        plt.yscale("log")

    plt.title("Pixel Value Distribution")
    plt.xlabel("Normalized VIL Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def nonzero_ratio_distribution(h5_path, max_events=None):
    """
    Calculates and plots the distribution of the ratio of non-zero pixels 
    per event in the HDF5 file.
    """
    ratios = []

    with h5py.File(h5_path, "r") as f:
        eids = list(f.keys())
        if max_events is not None:
            eids = eids[:max_events]

        for eid in tqdm(eids, desc="Non-zero ratio per event"):
            # Modification: Read data directly from f, replacing undefined read_vil
            vil = f[eid]["vil"][:]
            ratio = float((vil > 0).mean())
            ratios.append(ratio)

    ratios = np.array(ratios)
    plt.figure(figsize=(8,4))
    plt.hist(ratios, bins=50)
    plt.xlabel("Non-zero pixel ratio")
    plt.ylabel("Number of events")
    plt.title("Distribution of non-zero pixel ratio (per event)")
    plt.grid(alpha=0.3)
    plt.show()
    print("Non-zero ratio: mean =", ratios.mean(), "median =", np.median(ratios))


def plot_event_temporal_change(h5_path, event_idx=0):
    """
    Analyzes the temporal change intensity (mean absolute difference 
    between adjacent frames) for a single event.
    """
    with h5py.File(h5_path, "r") as f:
        eids = list(f.keys())
        eid = eids[event_idx]
        # Modification: Read data directly from f, replacing undefined read_vil
        vil = f[eid]["vil"][:].astype(np.float32)

    # Calculate absolute difference between consecutive frames: (H,W,T-1)
    d = np.abs(vil[:, :, 1:] - vil[:, :, :-1])        
    # Spatial mean of differences per time step: (T-1,)
    curve = d.mean(axis=(0,1))                        

    plt.figure(figsize=(10,4))
    plt.plot(np.arange(len(curve))+1, curve)
    plt.xlabel("time step")
    plt.ylabel("mean |Δ|")
    plt.title(f"Temporal change intensity (event={eid})")
    plt.grid(alpha=0.3)
    plt.show()


def compare_event_hists(h5_path, event_indices=(0, 100, 600), bins=256):
    """
    Compares the pixel value histograms of multiple specific events 
    to visualize variability in intensity distributions.
    """
    # Open the file
    with h5py.File(h5_path, "r") as f:
        eids = list(f.keys())
        chosen = []

        # Filter for valid indices
        for idx in event_indices:
            if 0 <= idx < len(eids):
                chosen.append(eids[idx])
            else:
                print(f"Warning: Index {idx} is out of bounds.")

        # Prepare plotting
        plt.figure(figsize=(10, 5))

        # Define bin edges from 0 to 256
        edges = np.arange(257)
        centers = (edges[:-1] + edges[1:]) / 2

        for eid in chosen:
            # Fix 1 & 3: Use the already open file 'f', do not call load_event_vil
            # Fix 2: Ensure data is read as an array, not a dictionary
            vil = f[eid]["vil"][:]

            # Compute histogram
            h, _ = np.histogram(vil, bins=edges)

            # Normalize to Probability Density Function (PDF) - avoid division by zero
            total_pixels = h.sum()
            if total_pixels > 0:
                h = h / total_pixels

            plt.plot(centers, h, label=f"Event {eid}", alpha=0.8)

        plt.yscale("log")
        plt.xlabel("Pixel value (VIL)")
        plt.ylabel("Normalized Frequency (log scale)")
        plt.title("Per-event Pixel Distributions (Sampled Events)")
        plt.grid(alpha=0.3, which="both") # which='both' makes log grid clearer
        plt.legend()
        plt.show()
