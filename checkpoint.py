"""
Checkpoint management functions.
"""

import os
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir='checkpoints'):
    """
    Save model checkpoint with optimizer and scheduler state.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (or None)
        epoch: Current epoch number
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: PyTorch optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load checkpoint to
    
    Returns:
        Tuple of (model, optimizer, scheduler, epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    model = model.to(device)
    print(f"✓ Checkpoint loaded from epoch {epoch}. Loss: {loss:.4f}")
    
    return model, optimizer, scheduler, epoch, loss
