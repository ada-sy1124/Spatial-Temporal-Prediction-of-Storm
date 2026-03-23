import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def create_filtered_h5(source_path, target_path, dataset_key="vil"):
    """
    Extracts a specific dataset key from a source H5 file into a new H5 file.
    
    Args:
        source_path: Path to the original H5 file.
        target_path: Path where the filtered file will be saved.
        dataset_key: The specific data key to extract from each event group.
    """
    # ensure output folder exists (e.g. "data/")
    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)

    with h5py.File(source_path, "r") as fin, h5py.File(target_path, "w") as fout:
        for eid in fin.keys():
            if dataset_key in fin[eid]:
                fout.create_group(eid)
                # Copy data slice into the new file structure
                fout[eid].create_dataset(dataset_key, data=fin[eid][dataset_key][:])

    return "Filtered file created."


# ==========================================
# 1. Dataset
# ==========================================
class VIL12to12Dataset(Dataset):
    """
    Custom Dataset for video-like sequences (VIL) from H5 files.
    Splits sequences into fixed-length input and output temporal segments.
    """
    def __init__(self, h5_path, index, in_len=12, out_len=12, dataset_key="vil"):
        self.h5_path = h5_path
        self.index = index  # List of (event_id, start_index) tuples
        self.in_len = in_len
        self.out_len = out_len
        self.dataset_key = dataset_key

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        eid, s = self.index[i]

        with h5py.File(self.h5_path, "r") as f:
            # Load specific slice across height, width, and time
            data = f[eid][self.dataset_key][:, :, s : s + self.in_len + self.out_len]

        # normalize to 0~1
        data = data.astype(np.float32) / 255.0

        # (H, W, T) -> (T, H, W)
        x_seq = np.transpose(data, (2, 0, 1))

        # Split into input/target and add channel dimension (C=1)
        x_in  = torch.from_numpy(x_seq[:self.in_len]).unsqueeze(1)      # (in_len, 1, H, W)
        x_out = torch.from_numpy(x_seq[self.in_len:]).unsqueeze(1)      # (out_len, 1, H, W)
        return x_in, x_out


# ==========================================
# 2. DataLoader builder
# ==========================================
def create_dataloaders(
    h5_path,
    num_starts,
    in_len,
    out_len,
    dataset_key="vil",
    batch_size=2,
    num_workers=0,
    test_size=0.2,
    seed=42,
    pin_memory=True,
):
    """
    Prepares training and validation DataLoaders by splitting unique H5 event IDs.
    
    Args:
        h5_path: Source H5 file path.
        num_starts: Number of sliding window starts to sample per event.
        test_size: Proportion of events to reserve for validation.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Error: {h5_path} not found!")

    print(f"Loading indices from {h5_path}...")

    with h5py.File(h5_path, "r") as f:
        all_ids = list(f.keys())

    if not all_ids:
        raise ValueError("Dataset is empty.")

    # Split unique event IDs to prevent data leakage between sets
    train_ids, val_ids = train_test_split(
        all_ids, test_size=test_size, random_state=seed, shuffle=True
    )

    print(f"Total Events: {len(all_ids)} | Train: {len(train_ids)} | Val: {len(val_ids)}")

    # Generate full index lists based on start positions
    train_index = [(eid, s) for eid in train_ids for s in range(num_starts)]
    val_index   = [(eid, s) for eid in val_ids   for s in range(num_starts)]

    train_dataset = VIL12to12Dataset(
        h5_path, train_index, in_len=in_len, out_len=out_len, dataset_key=dataset_key
    )
    val_dataset = VIL12to12Dataset(
        h5_path, val_index, in_len=in_len, out_len=out_len, dataset_key=dataset_key
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("DataLoaders ready.")
    return train_loader, val_loader