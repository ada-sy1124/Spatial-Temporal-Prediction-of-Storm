import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from config import CLASSES, cls2idx


def _read_one_modality(grp, key):
    # h5 stores (H, W, T) where T=36
    arr = grp[key][:]
    # -> (T, H, W)
    arr = np.moveaxis(arr, -1, 0)
    return arr


def load_storm(h5_path, sid, channels, target_size):
    """
    Return x: torch.float32 [T, C, target_size, target_size]
    """
    xs = []
    with h5py.File(h5_path, "r") as f:
        grp = f[sid]
        for k in channels:
            arr = _read_one_modality(grp, k)      # (T,H,W)
            x = torch.from_numpy(arr).float()     # [T,H,W]
            x = x.unsqueeze(1)                    # [T,1,H,W]
            x = F.interpolate(
                x,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
            x = x.squeeze(1)                      # [T,ts,ts]
            xs.append(x)

    x = torch.stack(xs, dim=1)                    # [T,C,ts,ts]
    return x


def estimate_mean_std(h5_path, storm_ids, channels, target_size, max_samples=200, seed=42):
    """
    - compute per-storm channel mean and mean(square), then average over storms
    """
    ids = list(storm_ids)[: min(max_samples, len(storm_ids))]

    ch_sum = torch.zeros(len(channels))
    ch_sumsq = torch.zeros(len(channels))
    n = 0

    for sid in ids:
        x = load_storm(h5_path, sid, channels, target_size)  # [T,C,H,W]
        v = x.permute(1, 0, 2, 3).reshape(len(channels), -1) # [C, N]
        ch_sum += v.mean(dim=1)
        ch_sumsq += (v ** 2).mean(dim=1)
        n += 1

    mean = ch_sum / max(n, 1)
    var = (ch_sumsq / max(n, 1)) - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-6))

    return mean.numpy().astype(np.float32), std.numpy().astype(np.float32)


def compute_class_weights_ipynb_style(train_df, mode="inv_sqrt", normalise=True):
    """
      - counts = bincount(y)
      - inv_sqrt: 1/sqrt(count)
    Returns: (counts, weights) both float32 arrays of shape (num_classes,)
    """
    y = train_df["event_type"].map(cls2idx()).values
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    counts = np.maximum(counts, 1.0)

    if mode == "inv_sqrt":
        w = 1.0 / np.sqrt(counts)
    elif mode == "inv":
        w = 1.0 / counts
    else:
        raise ValueError(f"Unknown class_weight_mode: {mode}")

    if normalise:
        w = w / np.mean(w)

    return counts.astype(np.float32), w.astype(np.float32)


class Task3Dataset(Dataset):
    """
    Returns (x, y, sid)
      x: [T, C, H, W]
      y: int64 scalar
      sid: string
    """
    def __init__(self, df_id, h5_path, channels, target_size, mean=None, std=None):
        self.df = df_id.reset_index(drop=True)
        self.h5_path = h5_path
        self.channels = channels
        self.target_size = target_size

        self.mean = None if mean is None else torch.tensor(mean).view(1, -1, 1, 1)  # [1,C,1,1]
        self.std = None if std is None else torch.tensor(std).view(1, -1, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["id"]
        y = int(cls2idx()[row["event_type"]])

        x = load_storm(self.h5_path, sid, self.channels, self.target_size)  # [T,C,H,W]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-6)

        return x, torch.tensor(y, dtype=torch.long), sid


def make_loaders(cfg, normalize=True, mean_std_max_samples=200):
    """
    Returns:
      train_loader, val_loader, stats
    stats includes:
      - mean/std (if normalize)
      - class_weights/counts computed on TRAIN split
    """
    df = pd.read_csv(cfg.events_csv, parse_dates=["start_utc"])
    df = df[["id", "event_type"]].copy()
    df = df[df["event_type"].isin(CLASSES)].reset_index(drop=True)

    # storm-level: one row per storm id
    df_id = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    train_df, val_df = train_test_split(
       df_id,
       test_size=0.2,
       random_state=42,
       stratify=df_id["event_type"],
    )


    if normalize:
        mean, std = estimate_mean_std(
            h5_path=cfg.train_h5,
            storm_ids=train_df["id"].tolist(),
            channels=cfg.channels,
            target_size=cfg.target_size,
            max_samples=mean_std_max_samples,
            seed=cfg.split_seed,
        )
    else:
        mean, std = None, None

    # ipynb-style class weights (train split)
    train_counts = class_weights = None
    if getattr(cfg, "use_class_weights", False):
        train_counts, class_weights = compute_class_weights_ipynb_style(
            train_df,
            mode=getattr(cfg, "class_weight_mode", "inv_sqrt"),
            normalise=getattr(cfg, "normalise_class_weights", True),
        )

    train_ds = Task3Dataset(train_df, cfg.train_h5, cfg.channels, cfg.target_size, mean=mean, std=std)
    val_ds = Task3Dataset(val_df, cfg.train_h5, cfg.channels, cfg.target_size, mean=mean, std=std)


    train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=0, 
    pin_memory=True,
    drop_last=True,
)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    stats = {
        "mean": mean,
        "std": std,
        "classes": CLASSES,
        "train_class_counts": train_counts,
        "class_weights": class_weights,
    }
    return train_loader, val_loader, stats
