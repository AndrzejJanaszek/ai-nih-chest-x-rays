"""
Data loading and Dataset class for chest X-ray images.
"""

import os
import glob
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from config import IMAGES_PATH, CSV_FILE_PATH, ALL_LABELS, WEIGHT_TYPE


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for Chest X-ray images.
    """
    
    def __init__(self, dataframe, image_paths_dict, labels_list, transform=None):
        """
        Args:
            dataframe: pandas DataFrame with image metadata
            image_paths_dict: Dictionary mapping image names to full paths
            labels_list: List of label column names
            transform: Optional image transformations
        """
        self.df = dataframe
        self.image_paths_dict = image_paths_dict
        self.labels_list = labels_list
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = self.image_paths_dict[img_name]
        image = Image.open(img_path).convert('RGB')

        # Extract binary values for diseases
        labels = self.df.iloc[idx][self.labels_list].values.astype('float32')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)


def calculate_sample_weights(df, labels_list, weight_type='sqrt'):
    """
    Calculate weights for each sample based on the rarest disease assigned to it.
    
    This ensures rare diseases appear more frequently in batches by weighting samples
    by their rarest disease.
    
    Args:
        df: pandas DataFrame with binary label columns
        labels_list: List of label names
        weight_type: 'sqrt' for square root, 'log' for logarithm
    
    Returns:
        numpy array of sample weights
    """
    # Calculate frequency of each disease
    disease_frequencies = {}
    for label in labels_list:
        disease_frequencies[label] = (df[label] == 1).sum()
    
    print("\nDisease frequencies:")
    for label, count in sorted(disease_frequencies.items(), key=lambda x: x[1]):
        print(f"  {label}: {count}")
    
    # Calculate inverse frequencies (weights for diseases)
    total_samples = len(df)
    disease_weights = {}
    for label in labels_list:
        freq = disease_frequencies[label]
        inv_freq = total_samples / max(freq, 1)  # Avoid division by zero
        
        # Apply transformation
        if weight_type == 'sqrt':
            disease_weights[label] = np.sqrt(inv_freq)
        elif weight_type == 'log':
            disease_weights[label] = np.log1p(inv_freq)  # log1p to avoid log(0)
        else:
            disease_weights[label] = inv_freq
    
    # For each sample, find the weight of its rarest disease
    sample_weights = np.ones(len(df), dtype=np.float64)
    
    for idx, row in df.iterrows():
        # Get all diseases assigned to this sample
        assigned_diseases = [label for label in labels_list if row[label] == 1]
        
        if assigned_diseases:
            # Use the weight of the rarest disease (max weight = rarest disease)
            max_weight = max(disease_weights[disease] for disease in assigned_diseases)
            sample_weights[idx] = max_weight
    
    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    print(f"\n✓ Sample weights calculated ({weight_type})")
    print(f"  Min weight: {sample_weights.min():.4f}")
    print(f"  Max weight: {sample_weights.max():.4f}")
    print(f"  Mean weight: {sample_weights.mean():.4f}")
    
    return sample_weights


def index_images(images_path):
    """
    Index all images in the rescaled_data directory.
    
    Args:
        images_path: Path to rescaled_data directory
    
    Returns:
        Dictionary mapping image names to full paths
    """
    print("Indexing images... (this may take a moment)")
    image_paths = {os.path.basename(x): x for x in glob.glob(
        os.path.join(images_path, 'images*', 'images', '*.png')
    )}
    print(f"✓ Found {len(image_paths)} images")
    return image_paths


def load_image_list(list_file):
    """
    Load list of image names from a text file.
    
    Args:
        list_file: Path to text file containing image names
    
    Returns:
        Set of image names
    """
    with open(list_file, 'r') as f:
        images = [line.strip() for line in f if line.strip()]
    return set(images)


def load_data_with_labels(csv_path, label_list):
    """
    Load CSV data and create binary columns for each label.
    
    Args:
        csv_path: Path to CSV file
        label_list: List of label names to create columns for
    
    Returns:
        pandas DataFrame with binary label columns
    """
    df = pd.read_csv(csv_path)
    
    # Create binary columns (0 or 1) for each disease
    for label in label_list:
        df[label] = df['Finding Labels'].map(lambda x: 1.0 if label in x else 0.0)
    
    print(f"✓ Loaded {len(df)} records from {csv_path}")
    return df


def create_datasets(image_paths_dict, df, train_images, val_images, train_transforms, val_transforms):
    """
    Create train and validation datasets.
    
    Args:
        image_paths_dict: Dictionary mapping image names to paths
        df: DataFrame with metadata
        train_images: Set of training image names
        val_images: Set of validation image names
        train_transforms: Transformations for training data
        val_transforms: Transformations for validation data
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Filter DataFrames based on image lists
    train_df = df[df['Image Index'].isin(train_images)].reset_index(drop=True)
    val_df = df[df['Image Index'].isin(val_images)].reset_index(drop=True)

    print(f"\nDataFrame after filtering:")
    print(f"  Training images: {len(train_df)}")
    print(f"  Validation images: {len(val_df)}")

    # Create datasets
    train_dataset = ChestXrayDataset(train_df, image_paths_dict, ALL_LABELS, transform=train_transforms)
    val_dataset = ChestXrayDataset(val_df, image_paths_dict, ALL_LABELS, transform=val_transforms)

    return train_dataset, val_dataset
