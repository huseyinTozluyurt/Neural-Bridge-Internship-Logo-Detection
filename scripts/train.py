"""
RF-DETR Logo Detection Training Script
Fine-tunes RF-DETR on logo detection dataset
"""

import os
import sys
import argparse
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATASET_DIR,
    CHECKPOINT_DIR,
    TRAINING_CONFIG,
    MODEL_CONFIG,
    WANDB_API_KEY
)

def train_rfdetr(
    variant="medium",
    dataset_dir=None,
    epochs=None,
    batch_size=None,
    lr=None,
    resume=None,
    output_dir=None,
    use_wandb=True,
    use_tensorboard=True,
    early_stopping=True,
    test_mode=False
):
    """
    Train RF-DETR model on logo dataset

    Args:
        variant: Model variant (small, medium, large, base)
        dataset_dir: Path to dataset directory
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        resume: Path to checkpoint to resume from
        output_dir: Directory to save checkpoints
        use_wandb: Enable Weights & Biases logging
        use_tensorboard: Enable TensorBoard logging
        early_stopping: Enable early stopping
        test_mode: Quick test with 2 epochs
    """

    # Use config defaults if not specified
    dataset_dir = dataset_dir or str(DATASET_DIR)
    output_dir = output_dir or str(CHECKPOINT_DIR)
    epochs = epochs or (2 if test_mode else TRAINING_CONFIG["epochs"])
    batch_size = batch_size or TRAINING_CONFIG["batch_size"]
    lr = lr or TRAINING_CONFIG["lr"]

    print("=" * 75)
    print("RF-DETR LOGO DETECTION TRAINING")
    print("=" * 75)
    print()
    print(f"Configuration:")
    print(f"  Model variant: {variant}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Gradient accumulation: {TRAINING_CONFIG['grad_accum_steps']}")
    print(f"  Mixed precision: {TRAINING_CONFIG['mixed_precision']}")
    print(f"  Early stopping: {early_stopping}")
    print(f"  TensorBoard: {use_tensorboard}")
    print(f"  W&B: {use_wandb}")
    if test_mode:
        print(f"\n  🧪 TEST MODE: Running quick test with {epochs} epochs")
    print()

    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ Warning: No GPU detected, training will be slow")
    print()

    # Verify dataset exists
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset directory not found: {dataset_dir}")
        print(f"   Please run: python scripts/download_dataset.py")
        return False

    # Check for required splits
    for split in ["train", "valid"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            print(f"❌ Error: {split} split not found: {split_dir}")
            return False

        images_dir = split_dir / "images"
        if not images_dir.exists():
            print(f"❌ Error: images directory not found: {images_dir}")
            return False

    print("✓ Dataset structure validated")
    print()

    # Import RF-DETR
    try:
        if variant.lower() == "small":
            from rfdetr import RFDETRSmall
            model = RFDETRSmall()
        elif variant.lower() == "medium":
            from rfdetr import RFDETRMedium
            model = RFDETRMedium()
        elif variant.lower() == "large":
            from rfdetr import RFDETRLarge
            model = RFDETRLarge()
        elif variant.lower() == "base":
            from rfdetr import RFDETRBase
            model = RFDETRBase()
        else:
            print(f"❌ Error: Unknown variant '{variant}'")
            print("   Supported: small, medium, large, base")
            return False

        print(f"✓ Loaded RF-DETR {variant.capitalize()} model")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    # Prepare training arguments
    train_args = {
        "dataset_dir": dataset_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": TRAINING_CONFIG["grad_accum_steps"],
        "lr": lr,
        "output_dir": output_dir,
        "early_stopping": early_stopping if not test_mode else False,
    }

    # Add W&B if enabled and API key is set
    if use_wandb and WANDB_API_KEY:
        train_args["wandb"] = True
        train_args["project"] = TRAINING_CONFIG["experiment_name"]
        train_args["run"] = f"{variant}_logo_detection"
        print("✓ W&B logging enabled")

    # Add TensorBoard if enabled
    if use_tensorboard:
        train_args["tensorboard"] = True
        print("✓ TensorBoard logging enabled")

    # Add resume checkpoint if specified
    if resume:
        train_args["resume"] = resume
        print(f"✓ Resuming from checkpoint: {resume}")

    print()
    print("=" * 75)
    print("STARTING TRAINING")
    print("=" * 75)
    print()

    try:
        # Start training
        model.train(**train_args)

        print()
        print("=" * 75)
        print("✅ TRAINING COMPLETE!")
        print("=" * 75)
        print()
        print(f"Checkpoints saved to: {output_dir}")
        print()

        if use_tensorboard:
            print("View training progress:")
            print(f"  tensorboard --logdir {output_dir}")

        print()
        print("Next steps:")
        print("  1. Evaluate model: python scripts/evaluate.py")
        print("  2. Run inference: python scripts/inference.py")
        print("  3. Export model: python scripts/export.py")

        return True

    except KeyboardInterrupt:
        print()
        print("⚠ Training interrupted by user")
        print(f"   Checkpoints saved to: {output_dir}")
        return False

    except Exception as e:
        print()
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train RF-DETR on logo detection dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training (medium model, 50 epochs)
  python scripts/train.py

  # Quick test (2 epochs)
  python scripts/train.py --test

  # Small model with custom settings
  python scripts/train.py --variant small --epochs 30 --batch-size 32

  # Resume from checkpoint
  python scripts/train.py --resume checkpoints/last.pth

  # Without W&B logging
  python scripts/train.py --no-wandb
        """
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=MODEL_CONFIG["variant"],
        choices=["small", "medium", "large", "base"],
        help="RF-DETR model variant"
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to dataset directory (default: from config.py)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config.py)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: from config.py)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: from config.py)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: from config.py)"
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging"
    )

    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )

    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode (2 epochs only)"
    )

    args = parser.parse_args()

    # Train the model
    success = train_rfdetr(
        variant=args.variant,
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        use_tensorboard=not args.no_tensorboard,
        early_stopping=not args.no_early_stopping,
        test_mode=args.test
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
