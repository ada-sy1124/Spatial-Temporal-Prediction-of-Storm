import pytest
import torch
import numpy as np
from types import SimpleNamespace
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import task3_script.train as training  # the file you posted

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def dummy_dataset():
    # 4 samples, 3 features each
    X = torch.randn(4, 3, 4, 4)  # pretend like (time, channels, H, W)
    y = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(X, y, torch.arange(4))  # last dummy for 'sid'
    return dataset

@pytest.fixture
def dummy_loader(dummy_dataset):
    return DataLoader(dummy_dataset, batch_size=2, shuffle=False)

@pytest.fixture
def dummy_model():
    # simple conv net to match input size
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(3*4*4, 2)
        def forward(self, x):
            return self.fc(self.flatten(x))
    return DummyNet()

@pytest.fixture
def dummy_cfg():
    return SimpleNamespace(
        lr=0.01,
        weight_decay=0.0,
        epochs=2,
        use_class_weights=True
    )

@pytest.fixture
def dummy_stats():
    return {
        "class_weights": np.array([1.0, 2.0], dtype=np.float32),
        "train_class_counts": np.array([2,2], dtype=np.int64)
    }

# -----------------------------
# Tests
# -----------------------------

def test_build_criterion_with_weights(dummy_cfg, dummy_stats):
    criterion = training.build_criterion(dummy_cfg, dummy_stats, device="cpu")
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.weight[0].item() == 1.0
    assert criterion.weight[1].item() == 2.0

def test_run_epoch_train_and_eval(dummy_model, dummy_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

    # train
    out_train = training._run_epoch(dummy_model, dummy_loader, criterion, optimizer, device="cpu", train=True)
    assert "loss" in out_train and "acc" in out_train and "f1_macro" in out_train

    # eval
    out_eval = training._run_epoch(dummy_model, dummy_loader, criterion, optimizer, device="cpu", train=False)
    assert "loss" in out_eval and "acc" in out_eval and "f1_macro" in out_eval

def test_fit_and_best_state(dummy_model, dummy_loader, dummy_cfg, dummy_stats):
    history, best_state = training.fit(dummy_model, dummy_loader, dummy_loader, dummy_cfg, device="cpu", stats=dummy_stats)
    assert isinstance(history, list)
    assert len(history) == dummy_cfg.epochs
    assert isinstance(best_state, dict)
    # check model state keys
    for k in dummy_model.state_dict().keys():
        assert k in best_state

def test_evaluate_with_cm(dummy_model, dummy_loader):
    metrics = training.evaluate_with_cm(dummy_model, dummy_loader, num_classes=2, device="cpu")
    assert "y_true" in metrics
    assert "y_pred" in metrics
    assert "cm" in metrics
    assert "acc" in metrics
    assert "f1_macro" in metrics
    assert metrics["cm"].shape == (2,2)
