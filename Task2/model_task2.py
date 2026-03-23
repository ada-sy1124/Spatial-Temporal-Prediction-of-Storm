# model.py
import torch
import torch.nn as nn
import os, base64
import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataloader_task2 import Task2WindowDataset, build_window_input_from_h5
from matplotlib.colors import PowerNorm


class SimpleED(nn.Module):
    def __init__(self, in_ch, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(),
        )
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(),
        )
        self.down2 = nn.MaxPool2d(2)

        self.bott = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.down1(x)
        x = self.enc2(x)
        x = self.down2(x)
        x = self.bott(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up1(x)
        x = self.dec1(x)
        return self.out(x)


@torch.no_grad()
def eval_epoch(model, loader, device="cpu"):
    model.eval()
    mse = nn.MSELoss()
    total_mse, total_mae, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        loss = mse(pred, y)
        mae = torch.mean(torch.abs(pred - y))

        bs = x.size(0)
        total_mse += loss.item() * bs
        total_mae += mae.item() * bs
        n += bs

    return total_mse / n, total_mae / n


def train_epoch(model, loader, optimizer, device="cpu"):
    model.train()
    mse = nn.MSELoss()
    total_mse, total_mae, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = mse(pred, y)
        mae = torch.mean(torch.abs(pred - y))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = x.size(0)
        total_mse += loss.item() * bs
        total_mae += mae.item() * bs
        n += bs

    return total_mse / n, total_mae / n


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=15, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)  # logits


class WeightedL1(nn.Module):
    def __init__(self, thr=0.1, alpha=4.0):
        super().__init__()
        self.thr = float(thr)
        self.alpha = float(alpha)

    def forward(self, pred, target):
        w = 1.0 + self.alpha * (target > self.thr).float()
        return (w * (pred - target).abs()).mean()


class WeightedL1Cont(nn.Module):
    def __init__(self, alpha=6.0, gamma=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, pred, target):
        w = 1.0 + self.alpha * (target**self.gamma)
        return (w * (pred - target).abs()).mean()


def gradient_loss(pred, target):
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


def make_criterion(
    mode="weighted", lam_grad=0.0, *, thr=0.1, alpha=4.0, alpha_cont=6.0, gamma=1.0
):
    if mode == "weighted":
        base = WeightedL1(thr=thr, alpha=alpha)
    elif mode == "cont":
        base = WeightedL1Cont(alpha=alpha_cont, gamma=gamma)
    else:
        raise ValueError("Unknown mode")

    def crit(pred, target):
        loss = base(pred, target)
        if lam_grad and lam_grad > 0:
            loss = loss + float(lam_grad) * gradient_loss(pred, target)
        return loss

    return crit


@torch.no_grad()
def eval_mae_mse(model, loader, device):
    model.eval()
    abs_sum01 = sq_sum01 = 0.0
    abs_sum255 = sq_sum255 = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = torch.sigmoid(model(x))

        err = pred - y
        abs_sum01 += err.abs().sum().item()
        sq_sum01 += (err**2).sum().item()

        err255 = (pred * 255.0) - (y * 255.0)
        abs_sum255 += err255.abs().sum().item()
        sq_sum255 += (err255**2).sum().item()

        n += err.numel()

    return {
        "mae01": abs_sum01 / n,
        "mse01": sq_sum01 / n,
        "mae255": abs_sum255 / n,
        "mse255": sq_sum255 / n,
    }


def save_ckpt(path, model, optimizer, epoch, train_hist, val_hist, best_val):
    ckpt = dict(
        model=model.state_dict(),
        opt=optimizer.state_dict(),
        epoch=epoch,
        train_hist=train_hist,
        val_hist=val_hist,
        best_val=best_val,
    )
    torch.save(ckpt, path)


def train_task2(
    train_ids,
    val_ids,
    *,
    cfg,
    device,
    dataset_cfg,
    loss_mode="weighted",
    lam_grad=0.0,
    loss_kwargs=None,
    ckpt_path="task2_best.pt",
):
    """
    cfg: dict with h5_path, K, batch_train, batch_val, num_workers, lr, epochs
    dataset_cfg: dict with T_TOTAL, H, W, RANGES, OFFSETS
    """
    loss_kwargs = loss_kwargs or {}

    model = UNetSmall(in_ch=15, out_ch=1, base=32).to(device)
    criterion = make_criterion(mode=loss_mode, lam_grad=lam_grad, **loss_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    train_ds = Task2WindowDataset(
        cfg["h5_path"],
        train_ids,
        mode="train",
        K=cfg["K"],
        T_TOTAL=dataset_cfg["T_TOTAL"],
        H=dataset_cfg["H"],
        W=dataset_cfg["W"],
        ranges=dataset_cfg["RANGES"],
        offsets=dataset_cfg["OFFSETS"],
    )
    val_ds = Task2WindowDataset(
        cfg["h5_path"],
        val_ids,
        mode="val",
        K=cfg["K"],
        T_TOTAL=dataset_cfg["T_TOTAL"],
        H=dataset_cfg["H"],
        W=dataset_cfg["W"],
        ranges=dataset_cfg["RANGES"],
        offsets=dataset_cfg["OFFSETS"],
    )

    # train_loader: workers>0 + persistent OK
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_train"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device == "cuda"),
        persistent_workers=(int(cfg["num_workers"]) > 0),
        prefetch_factor=(2 if int(cfg["num_workers"]) > 0 else None),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_val"]),
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    train_hist = {k: [] for k in ["mae01", "mse01", "mae255", "mse255"]}
    val_hist = {k: [] for k in ["mae01", "mse01", "mae255", "mse255"]}

    best_val = float("inf")
    best_state = None

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}", leave=True)
        run_loss = 0.0

        abs_sum01 = sq_sum01 = 0.0
        abs_sum255 = sq_sum255 = 0.0
        n_pix = 0

        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            pred01 = torch.sigmoid(model(x))
            loss = criterion(pred01, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            pbar.set_postfix(loss=run_loss / max(pbar.n, 1))

            err01 = pred01 - y
            abs_sum01 += err01.abs().sum().item()
            sq_sum01 += (err01**2).sum().item()

            err255 = (pred01 * 255.0) - (y * 255.0)
            abs_sum255 += err255.abs().sum().item()
            sq_sum255 += (err255**2).sum().item()

            n_pix += err01.numel()

        tr = {
            "mae01": abs_sum01 / max(n_pix, 1),
            "mse01": sq_sum01 / max(n_pix, 1),
            "mae255": abs_sum255 / max(n_pix, 1),
            "mse255": sq_sum255 / max(n_pix, 1),
        }

        va = eval_mae_mse(model, val_loader, device)

        for k in train_hist:
            train_hist[k].append(tr[k])
            val_hist[k].append(va[k])

        print(
            f"epoch {epoch:02d} | "
            f"train MAE {tr['mae01']:.4f} (≈{tr['mae255']:.2f}) | "
            f"val MAE {va['mae01']:.4f} (≈{va['mae255']:.2f})"
        )

        if va["mae01"] < best_val:
            best_val = va["mae01"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            save_ckpt(
                ckpt_path, model, optimizer, epoch, train_hist, val_hist, best_val
            )
            print(f"  -> wrote {ckpt_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best state. best val MAE01 =", best_val)

    train_ds.close()
    val_ds.close()
    return model, train_hist, val_hist


def plot_curves(train_hist, val_hist, use_255=False):
    epochs = np.arange(1, len(train_hist["mae01"]) + 1)
    mae_key = "mae255" if use_255 else "mae01"
    mse_key = "mse255" if use_255 else "mse01"

    plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, train_hist[mse_key], marker="o", label="Train MSE")
    ax1.plot(epochs, val_hist[mse_key], marker="o", label="Val MSE")
    ax1.set_title("MSE Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, train_hist[mae_key], marker="o", label="Train MAE")
    ax2.plot(epochs, val_hist[mae_key], marker="o", label="Val MAE")
    ax2.set_title("MAE Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.grid(True)
    ax2.legend()

    best_epoch = int(np.argmin(val_hist[mae_key])) + 1
    ax2.axvline(best_epoch, linestyle="--")
    ax2.text(best_epoch, min(val_hist[mae_key]), f"best={best_epoch}", va="bottom")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predict_task2_event(model, h5_path, eid, *, device, dataset_cfg):
    model.eval()
    H = dataset_cfg["H"]
    W = dataset_cfg["W"]
    offsets = dataset_cfg["OFFSETS"]
    ranges = dataset_cfg["RANGES"]

    with h5py.File(h5_path, "r") as f:
        g = f[eid]
        TT = g["vis"].shape[-1]
        out = np.zeros((H, W, TT), dtype=np.float32)

        for t in range(TT):
            x, _ = build_window_input_from_h5(
                g, t, H=H, W=W, ranges=ranges, offsets=offsets
            )
            xt = torch.from_numpy(x[None, ...]).to(device)
            pr01 = torch.sigmoid(model(xt))[0, 0].cpu().numpy()
            out[:, :, t] = np.clip(pr01 * 255.0, 0.0, 255.0).astype(np.float32)

    return out


def plot_pred_vs_gt(
    model, h5_path, eid, *, device, dataset_cfg, frames=(0, 10, 20, 35)
):
    pred = predict_task2_event(
        model, h5_path, eid, device=device, dataset_cfg=dataset_cfg
    )
    with h5py.File(h5_path, "r") as f:
        gt = f[eid]["vil"][:].astype(np.float32)

    for t in frames:
        gt_t = gt[:, :, t]
        pr_t = pred[:, :, t]
        err = np.abs(pr_t - gt_t)
        vmax_err = np.percentile(err, 99)

        fig, axs = plt.subplots(1, 3, figsize=(13, 4), dpi=120)
        norm_hi = PowerNorm(gamma=0.5, vmin=0, vmax=255)
        im0 = axs[0].imshow(gt_t, cmap="turbo", norm=norm_hi)
        axs[0].set_title(f"GT VIL t={t}")
        im1 = axs[1].imshow(pr_t, cmap="turbo", norm=norm_hi)
        axs[1].set_title("Pred VIL")
        im2 = axs[2].imshow(err, vmin=0, vmax=vmax_err, cmap="magma")
        axs[2].set_title("|Error| (p99)")

        for a in axs:
            a.axis("off")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()


def save_triptych_gif(
    model,
    h5_path,
    eid,
    *,
    device,
    dataset_cfg,
    out_gif="gt_pred_err.gif",
    fps=6,
    dpi=160,
):
    from IPython.display import Image, display

    pred = predict_task2_event(
        model, h5_path, eid, device=device, dataset_cfg=dataset_cfg
    )
    with h5py.File(h5_path, "r") as f:
        gt = f[eid]["vil"][:].astype(np.float32)

    frames = []
    TT = pred.shape[-1]

    for t in range(TT):
        gt_t = gt[:, :, t]
        pr_t = pred[:, :, t]
        er_t = np.abs(pr_t - gt_t)
        vmax_err = np.percentile(er_t, 99)

        fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), dpi=dpi)
        norm_hi = PowerNorm(gamma=0.5, vmin=0, vmax=255)
        axs[0].imshow(gt_t, cmap="turbo", norm=norm_hi)
        axs[0].set_title("GT")
        axs[1].imshow(pr_t, cmap="turbo", norm=norm_hi)
        axs[1].set_title("Pred")
        axs[2].imshow(er_t, vmin=0, vmax=vmax_err, cmap="magma")
        axs[2].set_title("|Err| (p99)")
        for a in axs:
            a.axis("off")
        fig.suptitle(f"{eid}  t={t}", y=1.02)
        plt.subplots_adjust(wspace=0.02)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(img)
        plt.close(fig)

    duration = 1.0 / fps
    imageio.mimsave(out_gif, frames, duration=duration, loop=0)
    display(Image(filename=out_gif))
    print("Saved:", out_gif)


def show_gif(path, width=900):
    from IPython.display import HTML, display

    assert os.path.exists(path), f"GIF not found: {path}"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    display(HTML(f'<img src="data:image/gif;base64,{data}" style="width:{width}px;">'))


def mae_on_mask(pred255, gt255, thr=20):
    m = gt255 > thr
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(pred255[m] - gt255[m])))


def quick_check(model, h5_path, eid, *, device, dataset_cfg, thr=20):
    pred = predict_task2_event(
        model, h5_path, eid, device=device, dataset_cfg=dataset_cfg
    )
    with h5py.File(h5_path, "r") as f:
        gt = f[eid]["vil"][:].astype(np.float32)

    overall = float(np.mean(np.abs(pred - gt)))
    masked = mae_on_mask(pred, gt, thr=thr)

    print("Overall MAE (0~255):", overall)
    print(f"Masked MAE (gt>{thr}) (0~255):", masked)
