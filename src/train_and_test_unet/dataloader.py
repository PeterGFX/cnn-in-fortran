import numpy as np
import pandas as pd
import xarray as xr
import torch
import os
from torch.utils.data import DataLoader, random_split
import torch
from dataset import T_2M_Dataset

def T_2M_preporcessor(t_arr, x_size=320, y_size=320):

    return t_arr[:,:x_size,:y_size]

def get_dataloaders(batch_size, in_len, out_len, 
                    val_split=200, data_dir="./data/cosmo_sample.zarr"):

    torch_dataset = T_2M_Dataset(
        root_dir=data_dir,
        in_len=in_len,
        out_len=out_len,
        transform=T_2M_preporcessor
    )
    dataset_len = len(torch_dataset)
    train_data, val_data = random_split(torch_dataset, 
                                        [dataset_len-val_split, val_split])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

if __name__ == '__main__':

    train_dataloader, _ = get_dataloaders(
        batch_size=1,
        in_len=5,
        out_len=5,
    )

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")