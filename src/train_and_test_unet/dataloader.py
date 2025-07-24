import numpy as np
import pandas as pd
import xarray as xr
import torch
import os
from torch.utils.data import DataLoader
import torch
from dataset import T_2M_Dataset

def T_2M_preporcessor(t_arr, x_size=320, y_size=320):

    return t_arr[:,:x_size,:y_size]

def get_dataloaders(batch_size, in_len, out_len):

    torch_dataset = T_2M_Dataset(
        root_dir="./data/cosmo_sample.zarr",
        in_len=in_len,
        out_len=out_len,
        transform=T_2M_preporcessor
    )

    train_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")