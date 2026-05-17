"""
Main training pipeline for DenseNet-121 model on Chest X-ray dataset.
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DEVICE, ALL_LABELS, NUM_CLASSES, PHASE1_LR, PHASE2_LR,
    PHASE1_CHECKPOINTS, PHASE2_CHECKPOINTS,
    TRAINING_VALIDATIONS_DIR, VALIDATION_THRESHOLDS_DIR,
    IMAGES_PATH, CSV_FILE_PATH, TRAIN_LIST_PATH, VAL_LIST_PATH,
    print_device_info
)
from data_loader import (
    index_images, load_image_list, load_data_with_labels, create_datasets
)
from transforms import get_train_transforms, get_val_transforms
from utils import (
    create_data_loaders, create_loss_function, create_optimizer,
    create_optimizer_for_classifier, create_scheduler, ensure_directories_exist
)
from model import (
    create_densenet121_model, freeze_feature_extractor, unfreeze_all_weights,
    get_class_weights
)
from training import train_model
from metrics import validate_model, validate_epoch_threshold_range, validate_and_plot_roc
from checkpoint import load_checkpoint


def main():
    """Main training pipeline"""

    print("=" * 60)
    print("DenseNet-121 Chest X-ray Classification")
    print("=" * 60)

    # Print device info
    print_device_info()

    # ============================================================
    # STEP 1: DATA LOADING AND PREPARATION
    # ============================================================
    print("\n[1/5] Loading and preparing data...")

    # Index all images
    all_image_paths = index_images(IMAGES_PATH)

    # Load CSV data with labels
    df = load_data_with_labels(CSV_FILE_PATH, ALL_LABELS)

    print(f"Total records in CSV: {len(df)}")
    print(f"Labels: {ALL_LABELS}")

    # Load train/val/test image lists
    train_images = load_image_list(TRAIN_LIST_PATH)
    val_images = load_image_list(VAL_LIST_PATH)

    print(f"\nImage lists loaded:")
    print(f"  Training: {len(train_images)}")
    print(f"  Validation: {len(val_images)}")

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        all_image_paths, df, train_images, val_images,
        get_train_transforms(), get_val_transforms()
    )

    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

    # ============================================================
    # STEP 2: MODEL INITIALIZATION
    # ============================================================
    print("\n[2/5] Initializing model...")

    model = create_densenet121_model(DEVICE, NUM_CLASSES)
    freeze_feature_extractor(model)

    # ============================================================
    # STEP 3: LOSS & OPTIMIZATION SETUP
    # ============================================================
    print("\n[3/5] Setting up loss and optimization...")

    # Calculate class weights
    print("\nClass weights:")
    pos_weights = get_class_weights(df, ALL_LABELS, DEVICE)

    # Create loss function
    criterion = create_loss_function(pos_weights, DEVICE)

    # ============================================================
    # STEP 4: TRAINING PHASE 1 (Optional - currently commented out)
    # ============================================================
    print("\n[4/5] Phase 1: Training frozen classifier head (optional)...")
    print("⚠ Phase 1 training is commented out. Uncomment in main() if desired.")

    # Uncomment below to run Phase 1 training:
    # optimizer_phase_1 = create_optimizer_for_classifier(model, PHASE1_LR)
    # ensure_directories_exist([PHASE1_CHECKPOINTS])
    # model = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     criterion=criterion,
    #     optimizer=optimizer_phase_1,
    #     device=DEVICE,
    #     scheduler=None,
    #     num_epochs=5,
    #     checkpoint_dir=PHASE1_CHECKPOINTS,
    #     validation_dir=os.path.join(TRAINING_VALIDATIONS_DIR, 'phase1'),
    #     start_epoch=0
    # )

    # ============================================================
    # STEP 5: TRAINING PHASE 2 (Optional - currently commented out)
    # ============================================================
    print("\n[5/5] Phase 2: Fine-tuning entire model (optional)...")
    print("⚠ Phase 2 training is commented out. Uncomment in main() if desired.")

    # Uncomment below to run Phase 2 training:
    # print("\nUnfreezing model for Phase 2...")
    # unfreeze_all_weights(model)
    # optimizer_phase_2 = create_optimizer(model, PHASE2_LR)
    # scheduler = create_scheduler(optimizer_phase_2)
    # ensure_directories_exist([PHASE2_CHECKPOINTS])
    # model = train_model(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     criterion=criterion,
    #     optimizer=optimizer_phase_2,
    #     device=DEVICE,
    #     scheduler=scheduler,
    #     num_epochs=20,
    #     checkpoint_dir=PHASE2_CHECKPOINTS,
    #     validation_dir=os.path.join(TRAINING_VALIDATIONS_DIR, 'phase2'),
    #     start_epoch=5
    # )

    # ============================================================
    # STEP 6: VALIDATION WITH ROC CURVE VISUALIZATION
    # ============================================================
    print("\n[*] Running validation and plotting ROC curves...")
    
    # Load model from Phase 2 checkpoint epoch 20
    print("\nLoading model from Phase 2 checkpoint (epoch 20)...")
    checkpoint_path = os.path.join(PHASE2_CHECKPOINTS, 'checkpoint_epoch_20.pt')
    model, _, _, epoch, loss = load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device=DEVICE)
    
    # Uncomment below to run threshold range validation:
    # ensure_directories_exist([VALIDATION_THRESHOLDS_DIR])
    # validate_epoch_threshold_range(
    #     model=model,
    #     val_loader=val_loader,
    #     criterion=criterion,
    #     device=DEVICE,
    #     epoch_num=20,
    #     checkpoint_dir=PHASE2_CHECKPOINTS,
    #     validation_dir=VALIDATION_THRESHOLDS_DIR,
    #     num_classes=NUM_CLASSES
    # )

    # Validate model and plot ROC curves
    results = validate_and_plot_roc(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        labels=ALL_LABELS,
        output_dir=VALIDATION_THRESHOLDS_DIR
    )

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS WITH ROC CURVES")
    print("=" * 60)
    print(f"Average Loss: {results['avg_loss']:.4f}")
    print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
    print(f"Hamming Loss: {results['hamming_loss']:.4f}")
    print(f"Precision (micro): {results['precision']:.4f}")
    print(f"Recall (micro): {results['recall']:.4f}")
    print(f"F1 Score (micro): {results['f1_score']:.4f}")
    print(f"Mean ROC AUC: {results['mean_roc_auc']:.4f}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
