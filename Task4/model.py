"""
Task 4 Model Definitions for Lightning Prediction.

This module contains all neural network models used for lightning prediction.
Models can be imported and used directly in notebooks:

    from task4.train import CNNLightningPredictor, DensityMapPredictor
    model = CNNLightningPredictor()

Available Models:
    - CNNLightningPredictor: Basic CNN for direct (t, x, y) prediction
    - DensityMapPredictor: 3D CNN encoder-decoder for density map prediction
    - DualCNNLightningPredictor: Dual-head architecture with spatial and temporal heads
    - TimeLightningModel2: 3D CNN for temporal lightning count prediction
    - LightningTimePredictor: 3D CNN for per-frame lightning prediction
    - LightningTimePredictor1: MLP-based time predictor from point positions
    - DensityTimePredictor: 3D CNN for joint density and time prediction
    - DensityTimeUNet: U-Net style architecture with skip connections
    - DualDecoderUNet: Shared encoder with separate density and time decoders
    - DualDecoderUNetV2: Improved dual decoder with temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CNNLightningPredictor - Basic CNN for direct coordinate prediction
# =============================================================================

class CNNLightningPredictor(nn.Module):
    """
    Basic CNN model for predicting fixed-length lightning strike sequences.

    Treats frames x channels as input channels and uses a CNN encoder
    followed by an MLP head to predict (t, x, y) coordinates.

    Args:
        input_channels: Number of input modalities (default: 4)
        num_frames: Number of temporal frames (default: 36)
        max_events: Maximum number of lightning events to predict (default: 50)

    Input: (B, channels, frames, H, W) - e.g., (B, 4, 36, 192, 192)
    Output: (B, max_events, 3) - normalized (t, x, y) in [0, 1]
    """
    def __init__(self, input_channels=4, num_frames=36, max_events=50):
        super().__init__()
        self.max_events = max_events

        in_channels = input_channels * num_frames

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_events * 3)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, x.shape[3], x.shape[4])

        features = self.encoder(x)
        predictions = self.head(features)
        predictions = torch.sigmoid(predictions)

        return predictions.reshape(batch_size, self.max_events, 3)


# =============================================================================
# DensityMapPredictor - 3D CNN encoder-decoder for density maps
# =============================================================================

class DensityMapPredictor(nn.Module):
    """
    3D CNN encoder-decoder for predicting spatial-temporal density maps.

    Uses 3D convolutions to capture spatial-temporal patterns and outputs
    a probability map of lightning locations for each time frame.

    Args:
        input_channels: Number of input modalities (default: 4)
        num_frames: Number of temporal frames (default: 36)

    Input: (B, channels, frames, H, W) - e.g., (B, 4, 36, 192, 192)
    Output: (B, frames, H, W) - density map in [0, 1]
    """
    def __init__(self, input_channels=4, num_frames=36):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 1, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

    def forward(self, x):
        features = self.encoder(x)
        density = self.decoder(features)
        return torch.sigmoid(density.squeeze(1))


# =============================================================================
# Dual-CNN Architecture Components
# =============================================================================

class SharedEncoder(nn.Module):
    """
    3D CNN encoder shared between spatial and temporal prediction heads.

    Input: (B, 4, 36, 192, 192)
    Output: (B, feature_dim) global feature vector
    """
    def __init__(self, in_channels=4, feature_dim=512):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 4, 4), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        features = self.conv_layers(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        return self.fc(pooled)


class SpatialHead(nn.Module):
    """
    Predicts (x, y) normalized coordinates for N strike slots.

    Input: (B, feature_dim)
    Output: (B, max_strikes, 2) - (x, y) in [0, 1]
    """
    def __init__(self, feature_dim=512, max_strikes=50000, hidden_dim=256):
        super().__init__()
        self.max_strikes = max_strikes

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_strikes * 2),
        )

    def forward(self, features):
        out = self.mlp(features)
        out = out.view(-1, self.max_strikes, 2)
        return torch.sigmoid(out)


class TemporalHead(nn.Module):
    """
    Predicts time (t) for N strike slots using 1D convolution.

    Input: (B, feature_dim)
    Output: (B, max_strikes, 1) - t in [0, 1]
    """
    def __init__(self, feature_dim=512, max_strikes=50000):
        super().__init__()
        self.max_strikes = max_strikes

        self.fc1 = nn.Linear(feature_dim, max_strikes * 16)

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, features):
        B = features.size(0)

        out = self.fc1(features)
        out = out.view(B, self.max_strikes, 16)
        out = out.permute(0, 2, 1)

        out = self.temporal_conv(out)
        out = out.permute(0, 2, 1)

        return torch.sigmoid(out)


class DualCNNLightningPredictor(nn.Module):
    """
    Dual-CNN model for predicting variable-length (t, x, y) coordinates.

    Architecture:
    - Shared 3D encoder extracts features from input
    - Spatial head predicts (x, y) for each strike slot
    - Temporal head predicts (t) for each strike slot
    - Stop head predicts when to stop outputting strikes

    Args:
        in_channels: Number of input modalities (default: 4)
        feature_dim: Dimension of encoded feature vector (default: 512)
        max_strikes: Maximum number of strikes to predict (default: 50000)

    Input: (B, channels, frames, H, W)
    Output: spatial_preds (B, N, 2), temporal_preds (B, N, 1), stop_preds (B, N)
    """
    def __init__(self, in_channels=4, feature_dim=512, max_strikes=50000):
        super().__init__()
        self.max_strikes = max_strikes

        self.encoder = SharedEncoder(in_channels, feature_dim)
        self.spatial_head = SpatialHead(feature_dim, max_strikes)
        self.temporal_head = TemporalHead(feature_dim, max_strikes)

        self.stop_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_strikes),
        )

    def forward(self, x, targets=None):
        features = self.encoder(x)

        spatial_preds = self.spatial_head(features)
        temporal_preds = self.temporal_head(features)
        stop_preds = self.stop_head(features)

        return spatial_preds, temporal_preds, stop_preds

    @torch.no_grad()
    def generate(self, x, max_strikes=None, stop_threshold=0.5):
        """Generate variable-length predictions."""
        self.eval()
        spatial_preds, temporal_preds, stop_preds = self.forward(x)
        stop_probs = torch.sigmoid(stop_preds)

        coords = torch.cat([temporal_preds, spatial_preds], dim=-1)

        results = []
        for i in range(coords.size(0)):
            stops = stop_probs[i] > stop_threshold
            if stops.any():
                n_strikes = stops.float().argmax().item()
                if n_strikes == 0:
                    n_strikes = 1
            else:
                n_strikes = self.max_strikes

            sample_coords = coords[i, :n_strikes]
            results.append(sample_coords.cpu())

        return results


# =============================================================================
# Time Prediction Models
# =============================================================================

class TimeLightningModel2(nn.Module):
    """
    3D CNN for predicting lightning counts per time bin.

    Uses spatiotemporal encoder with global spatial pooling to predict
    lightning activity for each temporal frame.

    Args:
        n_time_bins: Number of temporal bins to predict (default: 36)

    Input: (B, 4, 36, 192, 192)
    Output: (B, n_time_bins) - lightning count per time bin
    """
    def __init__(self, n_time_bins=36):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        self.spatial_pool = nn.AdaptiveAvgPool3d((n_time_bins, 1, 1))

        self.head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.spatial_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.head(x)
        return x.squeeze(1)


class LightningTimePredictor(nn.Module):
    """
    3D CNN for per-frame lightning prediction with careful weight initialization.

    Args:
        in_channels: Number of input modalities (default: 4)

    Input: (B, 4, 36, 192, 192)
    Output: (B, 36) - clamped log-scale predictions per frame
    """
    def __init__(self, in_channels=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((36, 2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = self.features(x)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, -1)

        out = self.fc(x)
        out = out.view(b, t)

        return torch.clamp(out, min=-10, max=10)


class LightningTimePredictor1(nn.Module):
    """
    MLP-based time predictor from point positions.

    Predicts temporal distribution of lightning from spatial coordinates
    using mean pooling over point features.

    Args:
        hidden_dim: Hidden dimension for MLP (default: 64)
        num_bins: Number of temporal bins to predict (default: 36)

    Input:
        positions: (B, N, 2) - (x, y) coordinates
        mask: (B, N) - 1 for valid points, 0 for padding
    Output: (B, num_bins) - lightning count per time bin
    """
    def __init__(self, hidden_dim=64, num_bins=36):
        super().__init__()
        self.mlp_point = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mlp_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins)
        )

    def forward(self, positions, mask):
        h = self.mlp_point(positions)
        h = h * mask.unsqueeze(-1)

        h = h.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)

        out = self.mlp_pool(h)
        out = F.relu(out)

        total_lightnings = mask.sum(dim=1, keepdim=True)
        out = out * total_lightnings / (total_lightnings.mean() + 1e-6)

        return out


# =============================================================================
# Density + Time Prediction Models
# =============================================================================

class DensityTimePredictor(nn.Module):
    """
    3D CNN for joint density and time prediction.

    Outputs 2 channels: spatial density and temporal information per pixel.

    Args:
        input_channels: Number of input modalities (default: 4)
        num_frames: Number of temporal frames (default: 36)

    Input: (B, 4, 36, 192, 192)
    Output: (B, 2, 36, 192, 192) - [density, time] in [0, 1]
    """
    def __init__(self, input_channels=4, num_frames=36):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return torch.sigmoid(out)


class DensityTimeUNet(nn.Module):
    """
    U-Net style 3D CNN with skip connections for density and time prediction.

    Uses trilinear upsampling + conv instead of transposed convolutions
    for smoother outputs.

    Args:
        input_channels: Number of input modalities (default: 4)

    Input: (B, 4, 36, 192, 192)
    Output: (B, 2, 36, 192, 192) - [density, time] in [0, 1]
    """
    def __init__(self, input_channels=4):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool3d((1, 2, 2))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )

        # Decoder with skip connections
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv3d(256 + 128, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv3d(128 + 64, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv3d(64 + 32, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        # Refinement
        self.refine = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        self.out_conv = nn.Conv3d(32, 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d1 = self.dec1(torch.cat([self.up1(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e1], dim=1))

        # Refinement
        d3 = self.refine(d3)

        return torch.sigmoid(self.out_conv(d3))


class DualDecoderUNet(nn.Module):
    """
    U-Net with shared encoder and separate decoders for density and time.

    Features lighter encoder (max 128 channels) with separate decoder paths
    optimized for each output type.

    Args:
        input_channels: Number of input modalities (default: 4)

    Input: (B, 4, 36, 192, 192)
    Output: (B, 2, 36, 192, 192) - [density, time] in [0, 1]
    """
    def __init__(self, input_channels=4):
        super().__init__()

        # Shared Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(input_channels, 24, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(24),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.enc2 = nn.Sequential(
            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.enc3 = nn.Sequential(
            nn.Conv3d(48, 96, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.bottleneck = nn.Sequential(
            nn.Conv3d(96, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        # Density Decoder
        self.density_up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.density_dec1 = nn.Sequential(
            nn.Conv3d(128 + 96, 96, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(),
        )
        self.density_up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.density_dec2 = nn.Sequential(
            nn.Conv3d(96 + 48, 48, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(48),
            nn.ReLU(),
        )
        self.density_up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.density_dec3 = nn.Sequential(
            nn.Conv3d(48 + 24, 24, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(),
        )
        self.density_out = nn.Conv3d(24, 1, kernel_size=1)

        # Time Decoder
        self.time_up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.time_dec1 = nn.Sequential(
            nn.Conv3d(128 + 96, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.time_up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.time_dec2 = nn.Sequential(
            nn.Conv3d(64 + 48, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.time_up3 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        self.time_dec3 = nn.Sequential(
            nn.Conv3d(32 + 24, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.time_out = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        # Shared encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        # Density decoder
        dd1 = self.density_dec1(torch.cat([self.density_up1(b), e3], dim=1))
        dd2 = self.density_dec2(torch.cat([self.density_up2(dd1), e2], dim=1))
        dd3 = self.density_dec3(torch.cat([self.density_up3(dd2), e1], dim=1))
        density = torch.sigmoid(self.density_out(dd3))

        # Time decoder
        td1 = self.time_dec1(torch.cat([self.time_up1(b), e3], dim=1))
        td2 = self.time_dec2(torch.cat([self.time_up2(td1), e2], dim=1))
        td3 = self.time_dec3(torch.cat([self.time_up3(td2), e1], dim=1))
        time_pred = torch.sigmoid(self.time_out(td3))

        return torch.cat([density, time_pred], dim=1)


class DualDecoderUNetV2(nn.Module):
    """
    Improved U-Net with shared encoder and separate decoders.

    Key improvements over V1:
    1. Shallower encoder: min resolution 48x48 (2 pooling stages)
    2. Non-bleeding time decoder: 1x1 convs in final layers
    3. Temporal modeling: 1D convs along time axis
    4. Sharper outputs: nearest upsampling + small kernels

    Args:
        input_channels: Number of input modalities (default: 4)

    Input: (B, 4, 36, 192, 192)
    Output: (B, 2, 36, 192, 192) - [density, time]
    """
    def __init__(self, input_channels=4):
        super().__init__()

        # Shared Encoder (2 pooling stages: 192 -> 96 -> 48)
        self.enc1 = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        # Density Decoder
        self.density_up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.density_dec1 = nn.Sequential(
            nn.Conv3d(128 + 64, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.density_up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.density_dec2 = nn.Sequential(
            nn.Conv3d(64 + 32, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        self.density_refine = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
        )
        self.density_out = nn.Conv3d(16, 1, kernel_size=1)

        # Time Decoder V2 (non-bleeding + temporal aware)
        self.time_up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.time_dec1 = nn.Sequential(
            nn.Conv3d(128 + 64, 48, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(48),
            nn.ReLU(),
        )

        self.time_up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.time_dec2 = nn.Sequential(
            nn.Conv3d(48 + 32, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        # Temporal modeling - convolutions along time axis only
        self.temporal_model = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        # Pointwise time regression (no spatial bleeding)
        self.time_head = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=1),
        )

    def forward(self, x):
        # Shared Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        # Density Decoder
        dd1 = self.density_dec1(torch.cat([self.density_up1(b), e2], dim=1))
        dd2 = self.density_dec2(torch.cat([self.density_up2(dd1), e1], dim=1))
        dd3 = self.density_refine(dd2)
        density = torch.relu(self.density_out(dd3))

        # Time Decoder V2
        td1 = self.time_dec1(torch.cat([self.time_up1(b), e2], dim=1))
        td2 = self.time_dec2(torch.cat([self.time_up2(td1), e1], dim=1))
        td3 = self.temporal_model(td2)
        time_pred = torch.relu(self.time_head(td3))

        return torch.cat([density, time_pred], dim=1)


# =============================================================================
# Model Registry for easy access
# =============================================================================

__all__ = [
    'CNNLightningPredictor',
    'DensityMapPredictor',
    'SharedEncoder',
    'SpatialHead',
    'TemporalHead',
    'DualCNNLightningPredictor',
    'TimeLightningModel2',
    'LightningTimePredictor',
    'LightningTimePredictor1',
    'DensityTimePredictor',
    'DensityTimeUNet',
    'DualDecoderUNet',
    'DualDecoderUNetV2',
]
