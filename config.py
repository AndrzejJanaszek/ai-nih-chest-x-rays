"""
Configuration file for the DenseNet-121 model training pipeline.
Contains all constants, paths, and hyperparameters.
"""

import os
import torch
from pathlib import Path

# ============================================================
# PATHS CONFIGURATION
# ============================================================

# Get the refactored_src directory (current directory)
REFACTORED_SRC_DIR = os.path.abspath(os.path.dirname(__file__))

# Get the main project root directory (parent of refactored_src)
PROJECT_ROOT = os.path.abspath(os.path.join(REFACTORED_SRC_DIR, '..'))

# Use refactored_src as the directory for checkpoints and training validations
SRC_DIR = REFACTORED_SRC_DIR

# Data paths (point to main project data folder)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
IMAGES_PATH = os.path.join(DATA_PATH, 'rescaled_data')
CSV_FILE_PATH = os.path.join(DATA_PATH, 'Data_Entry_2017.csv')

# Training list paths
TRAIN_LIST_PATH = os.path.join(DATA_PATH, 'train_list.txt')
VAL_LIST_PATH = os.path.join(DATA_PATH, 'validation_list.txt')
TEST_LIST_PATH = os.path.join(DATA_PATH, 'test_list.txt')

# Checkpoint paths
CHECKPOINTS_DIR = os.path.join(SRC_DIR, 'checkpoints')
PHASE1_CHECKPOINTS = os.path.join(CHECKPOINTS_DIR, 'phase1')
PHASE2_CHECKPOINTS = os.path.join(CHECKPOINTS_DIR, 'phase2')
FINAL_MODELS_DIR = os.path.join(CHECKPOINTS_DIR, 'final_models')

# Validation/Training results paths
TRAINING_VALIDATIONS_DIR = os.path.join(SRC_DIR, 'training_validations')
VALIDATION_THRESHOLDS_DIR = os.path.join(SRC_DIR, 'validation_thresholds')

# ============================================================
# FINAL MODEL SELECTION
# ============================================================

# Choose which final model variant to use
# Options: '14_labels', '14_labels_batch64', '14_labels_batch64_accum4', '8_labels', 14_label_batch64_accum4_log
FINAL_MODEL_VARIANT = '14_label_batch64_accum4_log'

# Epoch to load from the selected variant
FINAL_MODEL_EPOCH = 20

# ============================================================
# LABELS & CLASSES CONFIGURATION
# ============================================================

ALL_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration',
    'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia',
    'Pneumothorax', 'Hernia'
]
NUM_CLASSES = len(ALL_LABELS)

# ============================================================
# HYPERPARAMETERS
# ============================================================

# Training parameters
BATCH_SIZE = 64
NUM_WORKERS = 4  # Use 0 on Windows to avoid multiprocessing issues
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 20

# Learning rates
PHASE1_LR = 1e-3
PHASE2_LR = 1e-5

# Mixed Precision Training (AMP)
USE_AMP = True  # Enable Automatic Mixed Precision if CUDA is available

# Gradient Accumulation
GRADIENT_ACCUMULATION_STEPS = 4

# Weighted Sampling (balance rare diseases)
USE_WEIGHTED_SAMPLING = True  # Enable weighted sampling for rare diseases
WEIGHT_TYPE = 'log'  # 'sqrt' for square root, 'log' for logarithm

# Model parameters
IMAGE_SIZE = 224
RESIZE_SIZE = 256
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Validation thresholds
VALIDATION_THRESHOLDS = (0.0, 1.05, 0.05)  # (start, end, step)

# Custom disease-specific thresholds for GUI predictions
DISEASE_THRESHOLDS = {
    'Atelectasis': 0.4,
    'Cardiomegaly': 0.6,
    'Consolidation': 0.35,
    'Edema': 0.55,
    'Effusion': 0.5,
    'Emphysema': 0.35,
    'Fibrosis': 0.35,
    'Infiltration': 0.35,
    'Mass': 0.45,
    'Nodule': 0.35,
    'Pleural_Thickening': 0.3,
    'Pneumonia': 1.0,
    'Pneumothorax': 0.4,
    'Hernia': 0.85
}

# Dropout rate
DROPOUT_RATE = 0.3
HIDDEN_SIZE = 512

# ============================================================
# DEVICE CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_device_info():
    """Print information about the current device"""
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Running on CPU. Consider using GPU for faster training.")
