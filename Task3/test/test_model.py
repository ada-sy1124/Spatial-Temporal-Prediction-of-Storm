import pytest
import torch
from types import SimpleNamespace
import task3_script.model as models

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def dummy_input():
    # batch=2, time=3, channels=4, height=16, width=16
    return torch.randn(2, 3, 4, 16, 16)

@pytest.fixture
def dummy_cfg():
    return SimpleNamespace(
        channels=["ir", "vis", "radar", "sst"],
        num_classes=2,
        temporal_mode="mean",
        pretrained=False,
        feat_dim=64,
        dropout=0.1
    )

# -----------------------------
# Tests
# -----------------------------

@pytest.mark.parametrize("temporal_mode", ["mean", "meanmax", "lstm"])
def test_forward_pass_shapes(temporal_mode, dummy_input, dummy_cfg):
    dummy_cfg.temporal_mode = temporal_mode
    model = models.Task3ResNetTemporal(
        in_channels=len(dummy_cfg.channels),
        num_classes=dummy_cfg.num_classes,
        temporal_mode=temporal_mode,
        pretrained=False,
        feat_dim=dummy_cfg.feat_dim,
        dropout=dummy_cfg.dropout,
    )

    out = model(dummy_input)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (dummy_input.shape[0], dummy_cfg.num_classes)  # (B,num_classes)


def test_build_model_returns_model(dummy_cfg):
    model = models.build_model(dummy_cfg)
    assert isinstance(model, models.Task3ResNetTemporal)


def test_forward_lstm_output_dim(dummy_input):
    # specifically test lstm temporal mode
    model = models.Task3ResNetTemporal(
        in_channels=4,
        num_classes=3,
        temporal_mode="lstm",
        pretrained=False,
        feat_dim=32,
        dropout=0.0,
        lstm_hidden=16
    )
    out = model(dummy_input)
    assert out.shape == (dummy_input.shape[0], 3)


# -----------------------------
# Additional tests
# -----------------------------

def test_invalid_temporal_mode_raises(dummy_input):
    with pytest.raises(ValueError):
        models.Task3ResNetTemporal(
            in_channels=4,
            num_classes=2,
            temporal_mode="invalid_mode",
            pretrained=False,
            feat_dim=16
        )

def test_dropout_effect(dummy_input):
    model_no_dropout = models.Task3ResNetTemporal(
        in_channels=4, num_classes=2, temporal_mode="mean", dropout=0.0
    )
    model_dropout = models.Task3ResNetTemporal(
        in_channels=4, num_classes=2, temporal_mode="mean", dropout=0.9
    )
    out1 = model_no_dropout(dummy_input)
    out2 = model_dropout(dummy_input)
    # outputs should have same shape
    assert out1.shape == out2.shape

def test_feature_projection_dim(dummy_input):
    # Check that projection maps encoder output to correct feat_dim
    feat_dim = 128
    model = models.Task3ResNetTemporal(
        in_channels=4, num_classes=2, temporal_mode="mean", feat_dim=feat_dim
    )
    b, t, c, h, w = dummy_input.shape
    b_t = b * t
    feats = model.encoder(dummy_input.reshape(b_t, c, h, w))
    proj_feats = model.proj(feats)
    assert proj_feats.shape[1] == feat_dim

def test_temporal_meanmax_concat(dummy_input):
    model = models.Task3ResNetTemporal(
        in_channels=4, num_classes=2, temporal_mode="meanmax", feat_dim=32
    )
    out = model(dummy_input)
    # meanmax doubles feat_dim internally
    assert out.shape[1] == 2  # num_classes
