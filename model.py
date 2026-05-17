"""
Model initialization and creation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, HIDDEN_SIZE, DROPOUT_RATE


def create_densenet121_model(device, num_classes=NUM_CLASSES):
    """
    Create DenseNet-121 model with pretrained ImageNet weights.
    
    Args:
        device: Device to put model on
        num_classes: Number of output classes
    
    Returns:
        Model on specified device
    """
    # Load pretrained DenseNet-121
    weights = models.DenseNet121_Weights.IMAGENET1K_V1
    model = models.densenet121(weights=weights)

    # Replace classifier with custom head (before freezing!)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(HIDDEN_SIZE, num_classes)
    )

    # Move model to device
    model = model.to(device)

    print(f"✓ DenseNet-121 model created on {device}")
    return model


def freeze_feature_extractor(model):
    """
    Freeze feature extractor weights, keep classifier trainable (for Phase 1 training).
    
    Args:
        model: PyTorch model (DenseNet-121)
    """
    # Freeze all features (DenseNet-121 feature extractor)
    for param in model.features.parameters():
        param.requires_grad = False
    # Keep classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    print("✓ Feature extractor frozen, classifier trainable")


def unfreeze_all_weights(model):
    """
    Unfreeze all model weights (for Phase 2 fine-tuning).
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
    print("✓ All model weights unfrozen")


def get_class_weights(df, labels_list, device):
    """
    Calculate class weights for imbalanced dataset.
    
    Args:
        df: pandas DataFrame
        labels_list: List of label column names
        device: Device to put weights on
    
    Returns:
        Tensor of class weights
    """
    pos_weights = []
    
    for label in labels_list:
        num_pos = (df[label] == 1).sum()
        num_neg = (df[label] == 0).sum()
        weight = num_neg / num_pos if num_pos > 0 else 1.0
        weight = torch.sqrt(torch.tensor(weight))
        pos_weights.append(weight.item())
        print(f"{label:20s}: pos={num_pos:5d}, neg={num_neg:5d}, weight={weight.item():.4f}")

    pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    return pos_weights
