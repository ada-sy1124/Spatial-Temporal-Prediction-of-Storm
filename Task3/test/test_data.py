import numpy as np
import pandas as pd
import pytest
import torch
import h5py
from types import SimpleNamespace

import task3_script.data as dataset


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def dummy_h5(tmp_path):
    """
    Create a temporary h5 file with fake storm data
    """
    h5_path = tmp_path / "test.h5"

    # Create 6 storms: 3 for class A, 3 for class B
    for sid in ["storm_1", "storm_2", "storm_3", "storm_4", "storm_5", "storm_6"]:
        with h5py.File(h5_path, "a") as f:
            grp = f.create_group(sid)
            grp.create_dataset("ir", data=np.random.rand(8, 8, 36))
            grp.create_dataset("vis", data=np.random.rand(8, 8, 36))

    return str(h5_path)


@pytest.fixture
def dummy_df():
    # 3 samples per class to satisfy stratified split
    return pd.DataFrame({
        "id": ["storm_1", "storm_2", "storm_3", "storm_4", "storm_5", "storm_6"],
        "event_type": ["A", "A", "A", "B", "B", "B"]
    })



@pytest.fixture
def dummy_config(dummy_h5, tmp_path):
    return SimpleNamespace(
        events_csv=str(tmp_path / "events.csv"),
        train_h5=dummy_h5,
        channels=["ir", "vis"],
        target_size=16,
        batch_size=1,
        split_seed=42,
        use_class_weights=True,
        class_weight_mode="inv_sqrt",
        normalise_class_weights=True,
        test_size=2,  # set test size ≥ number of classes
    )


# -----------------------------
# Tests
# -----------------------------

@pytest.mark.parametrize("key", ["ir", "vis"])
def test_read_one_modality(dummy_h5, key):
    with h5py.File(dummy_h5, "r") as f:
        grp = f["storm_1"]
        arr = dataset._read_one_modality(grp, key)

    assert arr.shape == (36, 8, 8)
    assert isinstance(arr, np.ndarray)


def test_load_storm_shape(dummy_h5):
    x = dataset.load_storm(
        h5_path=dummy_h5,
        sid="storm_1",
        channels=["ir", "vis"],
        target_size=16,
    )

    assert isinstance(x, torch.Tensor)
    assert x.shape == (36, 2, 16, 16)
    assert x.dtype == torch.float32


@pytest.mark.parametrize("max_samples", [1, 2])
def test_estimate_mean_std(dummy_h5, max_samples):
    mean, std = dataset.estimate_mean_std(
        h5_path=dummy_h5,
        storm_ids=["storm_1", "storm_2", "storm_3", "storm_4"],
        channels=["ir", "vis"],
        target_size=16,
        max_samples=max_samples,
    )

    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(std > 0)


@pytest.mark.parametrize("mode", ["inv", "inv_sqrt"])
def test_compute_class_weights(mode, dummy_df, monkeypatch):
    monkeypatch.setattr(dataset, "CLASSES", ["A", "B"])
    monkeypatch.setattr(dataset, "cls2idx", lambda: {"A": 0, "B": 1})

    counts, weights = dataset.compute_class_weights_ipynb_style(
        dummy_df, mode=mode, normalise=True
    )

    assert counts.shape == (2,)
    assert weights.shape == (2,)
    assert np.isclose(weights.mean(), 1.0)


def test_task3_dataset_getitem(dummy_h5, dummy_df, monkeypatch):
    monkeypatch.setattr(dataset, "CLASSES", ["A", "B"])
    monkeypatch.setattr(dataset, "cls2idx", lambda: {"A": 0, "B": 1})

    ds = dataset.Task3Dataset(
        df_id=dummy_df,
        h5_path=dummy_h5,
        channels=["ir", "vis"],
        target_size=16,
        mean=np.array([0.5, 0.5]),
        std=np.array([0.2, 0.2]),
    )

    x, y, sid = ds[0]

    assert x.shape == (36, 2, 16, 16)
    assert isinstance(y, torch.Tensor)
    assert isinstance(sid, str)


def test_make_loaders_end_to_end(dummy_h5, dummy_df, dummy_config, tmp_path, monkeypatch):
    # write csv
    dummy_df.assign(start_utc=pd.Timestamp("2020-01-01")).to_csv(
        dummy_config.events_csv, index=False
    )

    monkeypatch.setattr(dataset, "CLASSES", ["A", "B"])
    monkeypatch.setattr(dataset, "cls2idx", lambda: {"A": 0, "B": 1})

    # explicitly set test_size to match stratified split requirements
    train_loader, val_loader, stats = dataset.make_loaders(
        dummy_config, normalize=True, mean_std_max_samples=2
    )

    # verify train + val dataset length
    assert len(train_loader.dataset) + len(val_loader.dataset) == len(dummy_df)
    assert "mean" in stats
    assert "std" in stats
    assert stats["class_weights"] is not None
