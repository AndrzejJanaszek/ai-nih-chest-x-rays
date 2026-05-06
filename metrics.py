"""
Validation and metrics computation.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    hamming_loss, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from checkpoint import load_checkpoint


def validate_model(model, val_loader, criterion, device, threshold):
    """
    Validate model on validation dataset with specified threshold.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation set
        criterion: Loss function
        device: Device to run on
        threshold: Decision threshold for binary classification
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    probabilities = []

    print(f"Validating (threshold={threshold})...")

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).cpu().numpy().astype(int)

            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy().astype(int))
            probabilities.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(val_loader.dataset)

    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    probabilities = np.array(probabilities)

    # Calculate metrics
    metrics = {
        'avg_loss': avg_loss,
        'exact_match_accuracy': accuracy_score(true_labels, predictions),
        'hamming_loss': hamming_loss(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='micro', zero_division=0),
        'recall': recall_score(true_labels, predictions, average='micro', zero_division=0),
        'f1_score': f1_score(true_labels, predictions, average='micro', zero_division=0)
    }

    # Calculate per-label metrics
    metrics['per_label_f1'] = f1_score(true_labels, predictions, average=None, zero_division=0).tolist()
    metrics['per_label_precision'] = precision_score(true_labels, predictions, average=None, zero_division=0).tolist()
    metrics['per_label_recall'] = recall_score(true_labels, predictions, average=None, zero_division=0).tolist()

    # Try to calculate AUC-ROC
    try:
        metrics['auc_roc'] = roc_auc_score(true_labels, probabilities, average='micro')
    except Exception:
        metrics['auc_roc'] = None

    print(f"✓ Validation completed. Avg Loss: {avg_loss:.4f}, F1 (micro): {metrics['f1_score']:.4f}")

    return metrics


def validate_epoch_threshold_range(model, val_loader, criterion, device, epoch_num, 
                                   checkpoint_dir, validation_dir, num_classes):
    """
    Validate model across multiple thresholds and save results.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation set
        criterion: Loss function
        device: Device to run on
        epoch_num: Epoch number to load checkpoint from
        checkpoint_dir: Directory containing checkpoints
        validation_dir: Directory to save validation results
        num_classes: Number of classes
    """
    print(">>> Threshold range validation -- START --")
    
    # Load checkpoint
    model, _, _, epoch, loss = load_checkpoint(
        checkpoint_path=os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch_num}.pt'),
        model=model,
        optimizer=None,
        scheduler=None,
        device=device
    )

    thresholds = np.arange(0.0, 1.05, 0.05)

    # Ensure validation directory exists
    os.makedirs(validation_dir, exist_ok=True)

    for threshold in thresholds:
        print(f"\n>>> Validating with threshold {threshold:.2f}...")
        val_metrics = validate_model(model, val_loader, criterion, device, threshold=threshold)

        # Save metrics to JSON
        val_metrics_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in val_metrics.items()
        }

        with open(os.path.join(validation_dir, f'validation_epoch_{epoch_num}_threshold_{threshold:.2f}.json'), 'w') as f:
            json.dump(val_metrics_serializable, f, indent=2)

    print(">>> Threshold range validation -- COMPLETED --")
