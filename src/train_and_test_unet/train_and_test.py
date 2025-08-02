
"""
PyTorch Script for Binary Classification using a Convolutional Neural Network
Includes:
- DataLoader setup (ImageFolder)
- Simple CNN definition
- Training and validation loops
- Model checkpointing
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models import UNet
from dataloader import get_dataloaders

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        #loss calculation
        running_loss += loss.item() * inputs.size(0) #loss * batch size
        #accuracy calculation
        # preds = torch.sigmoid(outputs) >= 0.5 
        # correct += (preds == targets.byte()).sum().item()

        total += inputs.size(0)

    epoch_loss = running_loss / total
    # epoch_acc = correct / total
    return epoch_loss#, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            #loss calcuation
            running_loss += loss.item() * inputs.size(0)
            # preds = torch.sigmoid(outputs) >= 0.5
            # correct += (preds == targets.byte()).sum().item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    # epoch_acc = correct / total
    return epoch_loss #, epoch_acc


def main(args):

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
 
    # Data loaders
    train_loader, val_loader = get_dataloaders(args.batch_size,
                                               in_len=args.in_len,
                                               out_len=args.out_len,
                                               data_dir=args.data_dir)

    # Model, loss, optimizer
    model = UNet(in_channels=args.in_len, 
                 num_classes=args.out_len, 
                 depth=5, merge_mode='concat').to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = 10e5
    os.makedirs(args.save_dir, exist_ok=True)

    print("Starting training...", flush=True)
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        sys.stdout.write(f"Epoch [{epoch}/{args.epochs}]"
              f" Train Loss: {train_loss:.4f}" #Train Acc: {train_acc:.4f}
              f" | Val Loss: {val_loss:.4f}\n") #Val Acc: {val_acc:.4f}
        sys.stdout.flush()

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(args.save_dir) / 'best_model.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}\n", flush=True)

    print("Training complete.", flush=True)
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Binary Classification CNN Training Script")
    parser.add_argument('--data-dir', type=str, default="/capstor/scratch/cscs/class172/cosmo_sample.zarr",
                        help="Root directory of dataset containing train/ and val/ folders")
    parser.add_argument('--save-dir', type=str, default='./checkpoint',
                        help="Directory to save model checkpoints")
    parser.add_argument('--in-len', type=int, default=5,
                        help="Batch size for training and validation")
    parser.add_argument('--out-len', type=int, default=5,
                        help="Batch size for training and validation")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for training and validation")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate for optimizer")
    args = parser.parse_args()
    main(args)
