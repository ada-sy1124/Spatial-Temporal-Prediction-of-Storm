from dataclasses import dataclass

# Task 3 event classes (keep this order consistent everywhere)
CLASSES = [
    "Flash Flood", "Flood", "Funnel Cloud", "Hail",
    "Heavy Rain", "Lightning", "Thunderstorm Wind", "Tornado"
]


@dataclass
class Task3Config:
    # Paths
    events_csv: str = "data/events.csv"
    train_h5: str = "data/train.h5"

    # Data
    target_size: int = 256
    out_t: int = 36
    channels: list | None = None  # filled in build_cfg()

    # Split
    val_size: float = 0.2
    split_seed: int = 42

    # Model (match your pure-ipynb baseline)
    temporal_mode: str = "mean"   # "mean" | "meanmax" | "lstm"
    pretrained: bool = False      # ipynb baseline: no pretrain
    num_classes: int = 8
    feat_dim: int = 256           # ipynb baseline: proj 512 -> 256
    dropout: float = 0.2          # ipynb baseline: dropout=0.2

    # Imbalance handling (match your pure-ipynb baseline)
    use_class_weights: bool = True
    class_weight_mode: str = "inv_sqrt"        # "inv" | "inv_sqrt"
    normalise_class_weights: bool = True       # divide by mean weight

    # Train (match your pure-ipynb baseline)
    epochs: int = 15
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4


def build_cfg(**kwargs) -> Task3Config:
    """Create config and fill defaults."""
    cfg = Task3Config(**kwargs)
    if cfg.channels is None:
        cfg.channels = ["vis", "ir069", "ir107", "vil"]
    return cfg


def cls2idx():
    return {c: i for i, c in enumerate(CLASSES)}
