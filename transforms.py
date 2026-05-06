"""
Image transformations for training, validation, and testing.
"""

from torchvision import transforms
from config import IMAGE_SIZE, RESIZE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD

# Training transformations with data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),  # Suitable for chest X-rays
    transforms.RandomRotation(10),       # Patient doesn't always stand perfectly straight
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
])

# Validation and test transformations (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
])

def get_train_transforms():
    """Get training transformations"""
    return train_transforms

def get_val_transforms():
    """Get validation/test transformations"""
    return val_transforms
