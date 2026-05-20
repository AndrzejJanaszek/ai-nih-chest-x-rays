"""
Utility functions for the training pipeline.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from config import BATCH_SIZE, NUM_WORKERS, DEVICE, USE_WEIGHTED_SAMPLING


def create_data_loaders(train_dataset, val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, 
                        device=DEVICE, train_weights=None):
    """
    Create train and validation data loaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        device: Device (for pin_memory setting)
        train_weights: Optional weights for training samples (for WeightedRandomSampler)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    pin_memory = device.type == 'cuda'

    # Create sampler for training data (use weighted sampler if weights provided)
    train_sampler = None
    train_shuffle = True
    
    if train_weights is not None and USE_WEIGHTED_SAMPLING:
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )
        train_shuffle = False
        print("✓ Using WeightedRandomSampler for training data")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=num_workers if num_workers > 0 else 0,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers > 0 else 0,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin_memory
    )

    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Validation loader: {len(val_loader)} batches")

    return train_loader, val_loader


def create_loss_function(pos_weights, device):
    """
    Create loss function with class weights.
    
    Args:
        pos_weights: Weights for positive class
        device: Device to put loss function on
    
    Returns:
        Loss function
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    print(f"✓ Loss function created (BCEWithLogitsLoss with class weights)")
    return criterion


def create_optimizer(model, learning_rate):
    """
    Create Adam optimizer for model.
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
    
    Returns:
        Optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"✓ Optimizer created (Adam, lr={learning_rate})")
    return optimizer


def create_optimizer_for_classifier(model, learning_rate):
    """
    Create optimizer for classifier only (Phase 1).
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
    
    Returns:
        Optimizer
    """
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print(f"✓ Optimizer created for classifier (Adam, lr={learning_rate})")
    return optimizer


def create_scheduler(optimizer, mode='min', factor=0.1, patience=3):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        mode: 'min' or 'max'
        factor: Factor to multiply learning rate by
        patience: Number of epochs with no improvement after which LR is reduced
    
    Returns:
        Learning rate scheduler
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience
    )
    print(f"✓ Scheduler created (ReduceLROnPlateau)")
    return scheduler


def ensure_directories_exist(directories):
    """
    Ensure all required directories exist.
    
    Args:
        directories: List of directory paths
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
