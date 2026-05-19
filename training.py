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
                validation_dir='training_validations', start_epoch=0, use_amp=True,
                accumulation_steps=1):
    """
    Train the model with optional Mixed Precision and Gradient Accumulation.
    
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
        accumulation_steps: Number of steps for gradient accumulation (default: 1)
    
    Returns:
        Trained model
    """
    print(f"STARTING TRAINING: {num_epochs} epochs")
    if accumulation_steps > 1:
        print(f"⚙ Using Gradient Accumulation: steps = {accumulation_steps}")
    
    # Enable AMP if available and requested
    use_amp = use_amp and torch.cuda.is_available() and device.type == 'cuda'
    if use_amp:
        print("✓ Using Mixed Precision (AMP)")
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None
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

        # Clear gradients BEFORE the batch loop
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # --- FORWARD PASS ---
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float())
                
                # Divide loss by number of accumulation steps to average gradients
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                loss = loss / accumulation_steps
                loss.backward()

            # --- OPTIMIZATION STEP (Every N steps) ---
            # Check if this is a step where we should update weights
            is_accumulation_boundary = (batch_idx + 1) % accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)

            if is_accumulation_boundary or is_last_batch:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Clear gradients AFTER weight update
                optimizer.zero_grad()

            # Restore original loss value for statistics (multiply back)
            original_loss_value = loss.item() * accumulation_steps
            running_loss += original_loss_value * inputs.size(0)
            pbar.set_postfix({'loss': f'{original_loss_value:.4f}'})

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
