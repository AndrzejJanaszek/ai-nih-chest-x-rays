"""
Model training functions.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from checkpoint import save_checkpoint
from metrics import validate_model


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                scheduler=None, num_epochs=5, checkpoint_dir='checkpoints',
                validation_dir='training_validations', start_epoch=0, use_amp=True):
    """
    Train the model for specified number of epochs with optional Mixed Precision.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of epochs to train
        checkpoint_dir: Directory to save checkpoints
        validation_dir: Directory to save validation results
        start_epoch: Starting epoch number (for resuming training)
        use_amp: Use Automatic Mixed Precision (default: True)
    
    Returns:
        Trained model
    """
    print(f"STARTING TRAINING: {num_epochs} epochs")
    
    # Enable AMP if available and requested
    use_amp = use_amp and torch.cuda.is_available() and device.type == 'cuda'
    if use_amp:
        print("✓ Using Mixed Precision (AMP)")
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None
        if use_amp:
            print("⚠ AMP requested but not available (CUDA not available)")
        else:
            print("Using standard precision training")

    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", total=len(train_loader))

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            if use_amp:
                # Forward pass with mixed precision
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float())
                
                # Backward pass with scaled loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # --- VALIDATION PHASE ---
        print(f"\n--- Validation Epoch {epoch+1} ---")
        val_results = validate_model(model, val_loader, criterion, device, threshold=0.5)
        val_loss = val_results.get('avg_loss')

        # --- UPDATE SCHEDULER ---
        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.8f}")

        # --- LOGGING & SAVING ---
        save_checkpoint(model, optimizer, scheduler, epoch+1, epoch_train_loss, checkpoint_dir)

        # Save JSON results
        val_results_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in val_results.items()
        }

        with open(os.path.join(validation_dir, f'validation_epoch_{epoch+1}.json'), 'w') as f:
            json.dump(val_results_serializable, f, indent=2)

    return model
