import torch
import torch.nn as nn
from torchvision import models as tv_models


class ResNet18Encoder(nn.Module):
    """Frame encoder: (B,C,H,W) -> (B,512)."""
    def __init__(self, in_channels=4, pretrained=False):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        m = tv_models.resnet18(weights=weights)

        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.feature = nn.Sequential(*list(m.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        return self.feature(x).flatten(1)  # (B,512)


class TemporalMean(nn.Module):
    def forward(self, feats):
        return feats.mean(dim=1)  # (B,T,D) -> (B,D)


class TemporalMeanMax(nn.Module):
    def forward(self, feats):
        mean = feats.mean(dim=1)
        mx, _ = feats.max(dim=1)
        return torch.cat([mean, mx], dim=1)  # (B,2D)


class TemporalLSTM(nn.Module):
    def __init__(self, feat_dim=256, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden_dim, batch_first=True)
        self.out_dim = hidden_dim

    def forward(self, feats):
        _, (h, _) = self.lstm(feats)
        return h[-1]  # (B,hidden_dim)


class Task3ResNetTemporal(nn.Module):
    """x: (B,T,C,H,W) -> logits: (B,num_classes)."""
    def __init__(
        self,
        in_channels,
        num_classes,
        temporal_mode="mean",
        pretrained=False,
        feat_dim=256,
        dropout=0.2,
        lstm_hidden=256,
    ):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels=in_channels, pretrained=pretrained)

        # ipynb baseline: proj 512 -> feat_dim
        self.proj = nn.Linear(self.encoder.out_dim, feat_dim)

        mode = temporal_mode.lower()
        if mode == "mean":
            self.temporal = TemporalMean()
            head_in = feat_dim
        elif mode == "meanmax":
            self.temporal = TemporalMeanMax()
            head_in = feat_dim * 2
        elif mode == "lstm":
            self.temporal = TemporalLSTM(feat_dim=feat_dim, hidden_dim=lstm_hidden)
            head_in = self.temporal.out_dim
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(head_in, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        f = self.encoder(x)         # (B*T,512)
        f = self.proj(f)            # (B*T,feat_dim)
        f = f.reshape(b, t, -1)     # (B,T,feat_dim)
        z = self.temporal(f)
        z = self.dropout(z)
        return self.head(z)


def build_model(cfg):
    in_channels = len(cfg.channels)
    return Task3ResNetTemporal(
        in_channels=in_channels,
        num_classes=cfg.num_classes,
        temporal_mode=cfg.temporal_mode,
        pretrained=cfg.pretrained,
        feat_dim=getattr(cfg, "feat_dim", 256),
        dropout=getattr(cfg, "dropout", 0.2),
    )
