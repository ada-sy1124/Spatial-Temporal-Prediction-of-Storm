from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

try:
    import h5py  # type: ignore
except Exception as e:  # pragma: no cover
    h5py = None  # noqa: N816

# seaborn is optional; we fall back to matplotlib if unavailable
try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None  # noqa: N816


# -----------------------------
# Generic training diagnostics
# -----------------------------

def plot_learning_curves(history, title: str = "Training Curves", save_path: Optional[str] = None):
    """
    Plot loss and macro-F1 curves from a history list of dicts produced by your engine.

    Expected keys in each history item:
      - epoch
      - train_loss, val_loss
      - train_f1_macro, val_f1_macro
    """
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    tr_f1 = [h["train_f1_macro"] for h in history]
    va_f1 = [h["val_f1_macro"] for h in history]

    plt.figure()
    plt.plot(epochs, tr_loss, label="train_loss")
    plt.plot(epochs, va_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title + " (Loss)")
    plt.legend()
    if save_path:
        plt.savefig(save_path.replace(".png", "_loss.png"), dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.plot(epochs, tr_f1, label="train_f1_macro")
    plt.plot(epochs, va_f1, label="val_f1_macro")
    plt.xlabel("epoch")
    plt.ylabel("macro-F1")
    plt.title(title + " (Macro-F1)")
    plt.legend()
    if save_path:
        plt.savefig(save_path.replace(".png", "_f1.png"), dpi=200, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """
    Simple confusion matrix plot

    Parameters
    ----------
    cm:
        Confusion matrix as (K, K).
    class_names:
        Names for each class, length K.
    normalize:
        If True, normalize each row to sum to 1.
    """
    cm = cm.astype(np.float32)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(title + (" (norm)" if normalize else ""))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            plt.text(j, i, txt, ha="center", va="center")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# -----------------------------
# Raw dataset loading (EDA)
# -----------------------------

DEFAULT_MODALITIES: Tuple[str, ...] = ("vis", "ir069", "ir107", "vil", "lght")


def load_event_raw(
    sid: str,
    h5_path: str = "data/train.h5",
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> Dict[str, np.ndarray]:
    """
    Load one storm event from the original H5 file.

    Notes
    -----
    - This mirrors eda_extend.ipynb: it returns raw arrays as stored in H5:
        images: (H, W, T)
        lght:   (N, 5)
    - For model input formatting (T, C, H, W) + resizing, use dataloader.load_storm.
    """
    if h5py is None:  # pragma: no cover
        raise ImportError("h5py is required for load_event_raw(). Please install h5py.")

    with h5py.File(h5_path, "r") as f:
        grp = f[sid]
        event = {k: grp[k][:] for k in modalities}
    return event


def temporal_profile_spatial_mean(x: np.ndarray) -> np.ndarray:
    """
    Compute temporal profile as spatial mean over frames.

    Parameters
    ----------
    x:
        Image cube stored as (H, W, T).

    Returns
    -------
    profile:
        (T,) float32 vector where profile[t] = mean over pixels at time t.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with ndim=3 (H,W,T). Got shape={x.shape}.")
    return x.mean(axis=(0, 1)).astype(np.float32)


@dataclass
class StormSummary:
    """
    Compact summary used for storm-level EDA (fast plotting after one pass).
    """
    id: str
    event_type: str

    # profiles for plotting
    vil_profile: np.ndarray
    ir069_profile: np.ndarray
    ir107_profile: np.ndarray
    vis_profile: np.ndarray

    # scalar features for boxplots
    vil_tmean: float
    vil_tmax: float
    vil_tstd: float

    ir069_tmean: float
    ir069_tmax: float
    ir069_tstd: float

    ir107_tmean: float
    ir107_tmax: float
    ir107_tstd: float

    vis_tmean: float
    vis_tmax: float
    vis_tstd: float
    vis_nonzero_frac: float


def summarise_event_raw(event: Mapping[str, np.ndarray], event_type: str, sid: str) -> StormSummary:
    """
    Reproduce the eda_extend.ipynb summarise_event() logic (raw arrays).
    """
    vil_pf = temporal_profile_spatial_mean(event["vil"])
    ir069_pf = temporal_profile_spatial_mean(event["ir069"])
    ir107_pf = temporal_profile_spatial_mean(event["ir107"])
    vis_pf = temporal_profile_spatial_mean(event["vis"])

    return StormSummary(
        id=sid,
        event_type=event_type,

        vil_profile=vil_pf,
        ir069_profile=ir069_pf,
        ir107_profile=ir107_pf,
        vis_profile=vis_pf,

        vil_tmean=float(vil_pf.mean()),
        vil_tmax=float(vil_pf.max()),
        vil_tstd=float(vil_pf.std()),

        ir069_tmean=float(ir069_pf.mean()),
        ir069_tmax=float(ir069_pf.max()),
        ir069_tstd=float(ir069_pf.std()),

        ir107_tmean=float(ir107_pf.mean()),
        ir107_tmax=float(ir107_pf.max()),
        ir107_tstd=float(ir107_pf.std()),

        vis_tmean=float(vis_pf.mean()),
        vis_tmax=float(vis_pf.max()),
        vis_tstd=float(vis_pf.std()),
        vis_nonzero_frac=float((event["vis"] != 0).mean()),
    )


def build_storm_eda_table(
    events_csv: str = "data/events.csv",
    train_h5: str = "data/train.h5",
    sample_n: Optional[int] = None,
    seed: int = 42,
    compute_time_of_day: bool = True,
    progress: bool = True,
):
    """
    Build a storm-level EDA table similar to eda_extend.ipynb.

    Returns a pandas DataFrame with one row per storm id, including:
      - event_type
      - temporal mean/max/std features per modality
      - profile arrays (vil_profile, ...)
      - (optional) hour_utc, hour_local and vis_tvar

    Parameters
    ----------
    sample_n:
        If set, randomly sample this many storms (useful for fast iteration).
    compute_time_of_day:
        Reproduces notebook's "local solar hour" approximation using storm longitude.
    """
    import pandas as pd  # local import to keep base deps light

    df = pd.read_csv(events_csv, parse_dates=["start_utc"])

    # storm-level meta (one row per id)
    cols = ["id", "event_type"]
    extra_cols = ["start_utc", "llcrnrlon", "urcrnrlon"]
    have_extras = all(c in df.columns for c in extra_cols)
    if compute_time_of_day and have_extras:
        meta = df.drop_duplicates("id")[cols + extra_cols].copy()
        meta["hour_utc"] = meta["start_utc"].dt.hour
        meta["lon_c"] = 0.5 * (meta["llcrnrlon"] + meta["urcrnrlon"])
        meta["hour_local"] = (meta["hour_utc"] + meta["lon_c"] / 15.0) % 24
        meta = meta[["id", "event_type", "hour_utc", "hour_local"]]
    else:
        meta = df.drop_duplicates("id")[cols].copy()

    if sample_n is not None:
        meta = meta.sample(n=min(sample_n, len(meta)), random_state=seed).reset_index(drop=True)
    else:
        meta = meta.reset_index(drop=True)

    rows = []
    it = meta.itertuples(index=False)
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore
            it = tqdm(list(it), desc="Summarising storms (EDA)")
        except Exception:
            it = meta.itertuples(index=False)

    for row in it:
        # row may include hour columns
        sid = getattr(row, "id")
        et = getattr(row, "event_type")
        event = load_event_raw(sid, h5_path=train_h5)
        s = summarise_event_raw(event, et, sid)
        d = s.__dict__.copy()
        # add hour info back if present
        if hasattr(row, "hour_utc"):
            d["hour_utc"] = getattr(row, "hour_utc")
        if hasattr(row, "hour_local"):
            d["hour_local"] = getattr(row, "hour_local")
        rows.append(d)

    storm_df = pd.DataFrame(rows)

    # extra derived feature from notebook
    if "vis_tstd" in storm_df.columns:
        storm_df["vis_tvar"] = storm_df["vis_tstd"] ** 2

    return storm_df


# -----------------------------
# EDA plotting helpers
# -----------------------------

def plot_class_distribution(
    storm_df,
    class_col: str = "event_type",
    title: str = "Class distribution (storm-level)",
    ax: Optional[plt.Axes] = None,
    rotate: int = 30,
):
    """
    Bar plot of storm counts by class.
    """
    counts = storm_df[class_col].value_counts()

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Number of storms")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=rotate)
    plt.tight_layout()
    return ax


def plot_boxplots_by_class(
    storm_df,
    cols: Sequence[Tuple[str, str]],
    class_col: str = "event_type",
    suptitle: str = "Feature by class",
    figsize: Tuple[int, int] = (12, 8),
    rotate: int = 30,
    use_seaborn: bool = True,
):
    """
    Create a 2x2 grid of boxplots by class for storm-level scalar features.

    Parameters
    ----------
    cols:
        List of (feature_col, title) pairs, typically length=4.
    use_seaborn:
        If seaborn is not installed, falls back to a matplotlib boxplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False)
    axes = axes.flat

    for ax, (feat, ttl) in zip(axes, cols):
        if use_seaborn and sns is not None:
            sns.boxplot(data=storm_df, x=class_col, y=feat, ax=ax)
        else:
            # matplotlib fallback: group into lists and draw boxplot
            groups = [g[feat].dropna().values for _, g in storm_df.groupby(class_col)]
            labels = [cls for cls, _ in storm_df.groupby(class_col)]
            ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_title(ttl, fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=rotate)

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_profiles_by_class(
    storm_df,
    profile_cols: Sequence[Tuple[str, str]],
    class_col: str = "event_type",
    q_lo: float = 25,
    q_hi: float = 75,
    sharex: bool = True,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot mean temporal profiles (spatial-mean over time) by class,
    with an interquartile band (q_lo..q_hi).

    profile_cols:
        List of (profile_column_name, display_title), typically 4 entries.
    """
    # assume all profiles have same length
    T = len(storm_df[profile_cols[0][0]].iloc[0])
    x = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=sharex)
    axes = axes.flat

    for ax, (col, ttl) in zip(axes, profile_cols):
        for cls, g in storm_df.groupby(class_col):
            profiles = np.stack(g[col].to_list(), axis=0)  # (N, T)
            mu = profiles.mean(axis=0)
            lo = np.percentile(profiles, q_lo, axis=0)
            hi = np.percentile(profiles, q_hi, axis=0)
            ax.plot(x, mu, linewidth=2, label=str(cls))
            ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_title(ttl)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Spatial mean")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=8, frameon=False)
    plt.tight_layout()
    return fig


def plot_vis_time_of_day(
    storm_df,
    time_col: str = "hour_local",
    mean_col: str = "vis_tmean",
    var_col: str = "vis_tvar",
    figsize: Tuple[int, int] = (10, 4),
):
    """
    Reproduce eda_extend.ipynb "VIS mean/var vs time-of-day" plots.

    Requires storm_df to contain time_col and the vis scalar features.
    """
    import pandas as pd  # local import

    req = {time_col, mean_col, var_col}
    missing = req - set(storm_df.columns)
    if missing:
        raise ValueError(f"storm_df missing columns required for this plot: {sorted(missing)}")

    agg = (
        storm_df
        .groupby(time_col)[[mean_col, var_col]]
        .mean()
        .reset_index()
        .sort_values(time_col)
    )

    # mean–time
    plt.figure(figsize=figsize)
    plt.plot(agg[time_col], agg[mean_col], marker="o")
    plt.xlabel(time_col)
    plt.ylabel("VIS temporal-mean (mean over T of spatial-mean)")
    plt.title("VIS mean vs time-of-day")
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # var–time
    plt.figure(figsize=figsize)
    plt.plot(agg[time_col], agg[var_col], marker="o")
    plt.xlabel(time_col)
    plt.ylabel("VIS temporal-variance (var over T of spatial-mean)")
    plt.title("VIS variance vs time-of-day")
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return agg
    # --- add to viz.py ---
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import h5py


def plot_multichannel_example(
    storm_id: str,
    train_h5: str = "data/train.h5",
    channels=("vis", "ir069", "ir107", "vil"),
    t_idx=(0, 12, 24),
    figsize=(10, 10),
    q_low=1,
    q_high=99,
    suptitle=True,
):
    """
    Plot a single storm across multiple channels at a few timesteps.

    Parameters
    ----------
    storm_id : str
        Storm/event id, e.g. "S778114"
    train_h5 : str
        Path to train.h5
    channels : tuple
        Channel names to plot (must exist under f[storm_id])
    t_idx : tuple
        Timesteps to plot (e.g. (0,12,24))
    figsize : tuple
        Matplotlib figure size
    q_low, q_high : int
        Percentiles for robust scaling (helps avoid all-black/all-white plots)
    suptitle : bool
        Add title.

    Returns
    -------
    fig, axes
    """

    def _robust_scale(frame2d: np.ndarray) -> np.ndarray:
        x = frame2d.astype(np.float32)
        lo, hi = np.percentile(x, [q_low, q_high])
        if hi <= lo:
            return x
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo)

    with h5py.File(train_h5, "r") as f:
        if storm_id not in f:
            raise KeyError(f"storm_id '{storm_id}' not found in {train_h5}")

        grp = f[storm_id]

        nrows, ncols = len(channels), len(t_idx)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = np.atleast_2d(axes)

        for r, ch in enumerate(channels):
            if ch not in grp:
                raise KeyError(f"channel '{ch}' not found under storm '{storm_id}'. Available: {list(grp.keys())}")

            arr = grp[ch][:]  # expected (H, W, T)
            if arr.ndim != 3:
                raise ValueError(f"Expected (H,W,T) for channel '{ch}', got shape {arr.shape}")

            T = arr.shape[-1]
            for c, t in enumerate(t_idx):
                if t < 0 or t >= T:
                    raise IndexError(f"t={t} out of range for channel '{ch}' with T={T}")

                ax = axes[r, c]
                frame = _robust_scale(arr[:, :, t])
                ax.imshow(frame)
                ax.set_xticks([])
                ax.set_yticks([])

                if r == 0:
                    ax.set_title(f"t={t}")
                if c == 0:
                    ax.set_ylabel(ch, rotation=0, labelpad=30)

    if suptitle:
        fig.suptitle(f"Multi-channel example storm: {storm_id}", y=0.92)

    fig.tight_layout()
    return fig, axes
