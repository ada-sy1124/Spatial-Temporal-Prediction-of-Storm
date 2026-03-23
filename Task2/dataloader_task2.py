import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class EventStore:
    def __init__(
        self, h5_path="data/train.h5", keys=("vis", "ir069", "ir107", "vil", "lght")
    ):
        self.h5_path = h5_path
        self.keys = keys
        self.f = None

    def __enter__(self):
        self.f = h5py.File(self.h5_path, "r")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.f.close()

    def get(self, event_id):
        g = self.f[event_id]
        return {k: g[k][:] for k in self.keys}


def resize_ir_to_vil(ir_img, target_shape=(384, 384)):
    """
    Resize IR images (192x192) to VIL grid (384x384).

    Supports:
      - Single frame: (H, W)
      - Time series:  (H, W, T)

    Returns:
      - Resized array with shape (384, 384) or (384, 384, T)
    """
    if ir_img.ndim == 2:
        # Single frame
        return cv2.resize(
            ir_img,
            target_shape[::-1],  # (width, height)
            interpolation=cv2.INTER_NEAREST,
        )

    elif ir_img.ndim == 3:
        # Time series
        T = ir_img.shape[-1]
        out = np.zeros((*target_shape, T), dtype=ir_img.dtype)

        for t in range(T):
            out[:, :, t] = cv2.resize(
                ir_img[:, :, t], target_shape[::-1], interpolation=cv2.INTER_NEAREST
            )
        return out

    else:
        raise ValueError(f"Unsupported IR shape: {ir_img.shape}")


def to_float32(x, fill_value=0.0):
    """
    Convert array to float32.
    Replace NaN / Inf with fill_value if input is floating type.
    """
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        x = np.where(np.isfinite(x), x, fill_value)

    return x.astype(np.float32, copy=False)


def norm_clip_to_m11(x, vmin, vmax):
    """clip to [vmin,vmax] then scale to [-1,1]"""
    x = x.astype(np.float32)
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin) * 2.0 - 1.0
    return x


def resize_ir_frame_to_hw(ir_192, *, H, W):
    """(192,192) -> (H,W) bilinear"""
    return cv2.resize(ir_192, (W, H), interpolation=cv2.INTER_LINEAR)


def clamp_t(t, T):
    return max(0, min(T - 1, int(t)))


def build_window_input_from_h5(g, t, *, H, W, ranges, offsets):
    """
    g: h5 group, contains vis/ir069/ir107/vil
    t: current time index
    returns:
      x: (3*len(offsets), H, W) float32 in [-1,1]
      y: (1, H, W) float32 in [0,1]
    """
    frames = []
    TT = g["vis"].shape[-1]

    for dt in offsets:
        ti = clamp_t(t + dt, TT)

        vis = g["vis"][:, :, ti]  # (H,W)
        ir069 = resize_ir_frame_to_hw(g["ir069"][:, :, ti], H=H, W=W)
        ir107 = resize_ir_frame_to_hw(g["ir107"][:, :, ti], H=H, W=W)

        vis = norm_clip_to_m11(vis, *ranges["vis"])
        ir069 = norm_clip_to_m11(ir069, *ranges["ir069"])
        ir107 = norm_clip_to_m11(ir107, *ranges["ir107"])

        frames += [vis, ir069, ir107]

    x = np.stack(frames, axis=0).astype(np.float32)

    vil_t = clamp_t(t, g["vil"].shape[-1])
    vil = g["vil"][:, :, vil_t].astype(np.float32) / 255.0
    y = vil[None, ...].astype(np.float32)
    return x, y


class Task2WindowDataset(Dataset):
    def __init__(
        self, h5_path, ids, mode="train", K=8, *, T_TOTAL, H, W, ranges, offsets
    ):
        self.h5_path = h5_path
        self.ids = list(ids)
        self.mode = mode
        self.T_TOTAL = int(T_TOTAL)
        self.K = int(K)

        self.H = int(H)
        self.W = int(W)
        self.ranges = ranges
        self.offsets = offsets

        self._h5 = None

        if mode == "train":
            self.frames_per_event = self.K
        else:
            self.frames_per_event = self.T_TOTAL

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.ids) * self.frames_per_event

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r", swmr=True) as h5:
            eid = self.ids[idx // self.frames_per_event]

            if self.mode == "train":
                t = np.random.randint(0, self.T_TOTAL)
            else:
                t = int(idx % self.T_TOTAL)

            g = h5[eid]
            x, y = build_window_input_from_h5(
                g,
                t,
                H=self.H,
                W=self.W,
                ranges=self.ranges,
                offsets=self.offsets,
            )
        return torch.from_numpy(x), torch.from_numpy(y)

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None
