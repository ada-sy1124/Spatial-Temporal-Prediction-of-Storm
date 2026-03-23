import pandas as pd
import h5py
import IPython
import PIL
import matplotlib.pyplot as plt
import os
import io
import numpy as np

def fetch_data():
    # IMPORTANT:
    #   Whenever you start a new colab runtime, use the following code to download
    #   the training dataset onto the runtime local storage.
    #   This should take ~3-5 mins for the whole dataset.
    #   You can then load data from the local storage (/content/data) into your colab
    #   notebook using the `h5py` library (see example below).
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="benmoseley/ese-dl-2025-26-group-project", filename="train.h5", repo_type="dataset", local_dir="data")
    hf_hub_download(repo_id="benmoseley/ese-dl-2025-26-group-project", filename="events.csv", repo_type="dataset", local_dir="data")


# this function loads all of the image arrays for a given id
def load_event(id):
    "Load event"
    with h5py.File(f'data/train.h5','r') as f:
        event = {img_type: f[id][img_type][:] for img_type in ['vis', 'ir069', 'ir107', 'vil', 'lght']}
    return event