"""
Validation and metrics computation.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    hamming_loss, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, auc
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


def validate_and_plot_roc(model, val_loader, criterion, device, labels, output_dir=None):
    """
    Validate model and plot ROC curves for all classes.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation set
        criterion: Loss function
        device: Device to run on
        labels: List of label names
        output_dir: Directory to save ROC curve plot (optional)
    
    Returns:
        Dictionary with metrics and ROC data
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    probabilities = []

    print("Computing ROC curves (collecting predictions)...")

    with torch.no_grad():
        for inputs, labels_batch in tqdm(val_loader, desc="Validation"):
            inputs, labels_batch = inputs.to(device), labels_batch.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels_batch.float())

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.3).cpu().numpy().astype(int)

            predictions.extend(preds)
            true_labels.extend(labels_batch.cpu().numpy().astype(int))
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

    # Plot ROC curves
    print("\nPlotting ROC curves...")
    num_classes = len(labels)
    
    # Calculate grid dimensions dynamically
    cols = 4
    rows = (num_classes + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(true_labels[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        
        fpr_dict[i] = fpr.tolist()
        tpr_dict[i] = tpr.tolist()
        roc_auc_dict[i] = roc_auc
        
        axes[i].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'ROC Curve - {labels[i]}')
        axes[i].legend(loc="lower right")
        axes[i].grid(alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'roc_curves.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ ROC curves saved to: {output_path}")
    
    plt.show()
    
    metrics['roc_auc_per_class'] = roc_auc_dict
    metrics['mean_roc_auc'] = np.mean(list(roc_auc_dict.values()))
    
    print(f"\n✓ Mean ROC AUC: {metrics['mean_roc_auc']:.4f}")
    print("\nPer-class ROC AUC:")
    for i, label in enumerate(labels):
        print(f"  {label}: {roc_auc_dict[i]:.4f}")
    
    return metrics
