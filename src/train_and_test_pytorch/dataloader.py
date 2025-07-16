from configs import cfg
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(data_dir, batch_size, img_size):
    """
    Create training and validation DataLoaders using ImageFolder.
    Directory structure:
        data_dir/
            train/
                class0/
                class1/
            val/
                class0/
                class1/
    """
    # Define transforms for training and validation
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
