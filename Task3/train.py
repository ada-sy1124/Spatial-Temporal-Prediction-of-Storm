import random
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # determinism like many ipynb baselines
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_criterion(cfg, stats, device):
    """
    Weighted CrossEntropy + weights provided by dataloader (train split).
    """
    cw = None if stats is None else stats.get("class_weights", None)
    if getattr(cfg, "use_class_weights", False) and cw is not None:
        w = torch.tensor(cw, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()


def _run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()

    losses, ys, preds = [], [], []

    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        pred = logits.argmax(dim=1)

        ys.append(yb.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(preds) if preds else np.array([])

    return {
        "loss": float(np.mean(losses)) if len(losses) else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else float("nan"),
    }


def fit(model, train_loader, val_loader, cfg, device, stats=None):
    """
    Train loop wrapper
    Returns: 
      history(list of dict), best_state_dict(cpu tensors)
    """
    criterion = build_criterion(cfg, stats, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    best_f1 = -1.0
    best_state = None

    if getattr(cfg, "use_class_weights", False) and stats is not None and stats.get("class_weights", None) is not None:
        print("train class counts:", stats.get("train_class_counts"))
        print("class weights:", stats.get("class_weights"))

    for epoch in range(1, cfg.epochs + 1):
        tr = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)

        if va["f1_macro"] > best_f1:
            best_f1 = va["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.3f} f1 {tr['f1_macro']:.3f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.3f} f1 {va['f1_macro']:.3f}"
        )

    return history, best_state


@torch.no_grad()
def evaluate_with_cm(model, loader, num_classes, device):
    """
    Returns: dict with y_true, y_pred, cm, acc, f1_macro
    """
    model.eval()
    ys, preds = [], []

    for xb, yb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)

        ys.append(yb.cpu().numpy())
        preds.append(pred.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "cm": cm,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
