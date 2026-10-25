# Configuration file for RF-DETR Logo Detection System

import os
from pathlib import Path

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "logos"
CHECKPOINT_DIR = PROJECT_ROOT / "rf_detr_logo_checkpoints"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys (load from environment or .env file)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

# Model Configuration
MODEL_CONFIG = {
    "variant": "medium",  # Options: "small", "medium", "large"
    "pretrained": True,
    "checkpoint_path": None,  # Path to custom checkpoint if resuming
}

# Training Configuration
TRAINING_CONFIG = {
    # Dataset
    "dataset_dir": str(DATASET_DIR),
    
    # Training hyperparameters
    "epochs": 50,
    "batch_size": 16,  # Adjust based on GPU memory
    "grad_accum_steps": 1,  # Effective batch = batch_size * grad_accum_steps
    "lr": 1e-4,
    "lr_encoder": 1e-5,  # Lower LR for encoder (optional)
    "weight_decay": 1e-4,
    
    # Model input
    "resolution": 576,  # Must be divisible by 56; common values: 384, 576, 800
    
    # Optimization
    "use_ema": True,  # Exponential Moving Average
    "ema_decay": 0.9999,
    "gradient_checkpointing": False,  # Enable if OOM
    "mixed_precision": True,  # FP16 training
    
    # Regularization
    "early_stopping": True,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.001,
    
    # Logging
    "log_interval": 10,  # Log every N steps
    "eval_interval": 1,  # Evaluate every N epochs
    "save_interval": 5,  # Save checkpoint every N epochs
    "wandb": True,  # Enable W&B logging
    "tensorboard": True,  # Enable TensorBoard logging
    
    # Output
    "output_dir": str(CHECKPOINT_DIR),
    "experiment_name": "logo_detection_rfdetr",
}

# Inference Configuration
INFERENCE_CONFIG = {
    "threshold": 0.5,  # Confidence threshold for detections
    "nms_threshold": 0.45,  # NMS IoU threshold (if applicable)
    "max_detections": 100,  # Max detections per image
    "device": "cuda",  # "cuda" or "cpu"
}

# CLIP Configuration (for Phase 4)
CLIP_CONFIG = {
    "model": "ViT-B/32",  # Options: "ViT-B/32", "ViT-B/16", "ViT-L/14"
    "device": "cuda",
    "batch_size": 32,  # For batch processing crops
    "threshold": 0.5,  # Minimum similarity score
    
    # Brand prompts (customize based on your dataset)
    "brand_prompts": [
        "a Nike swoosh logo",
        "an Adidas three stripes logo", 
        "a Coca-Cola script logo",
        "an Apple logo with bitten apple",
        "a McDonald's golden arches logo",
        "a Starbucks green circular logo",
        "a Pepsi circular logo",
        "an Amazon smile logo",
        "a Google colorful text logo",
        "a Microsoft four-square logo",
        # Add more as needed
    ],
}

# Dataset Configuration
DATASET_CONFIG = {
    # Dataset source
    "source": "roboflow",  # Options: "roboflow", "local", "logodet3k", "flickrlogos32"
    "roboflow_workspace": "",  # Fill in your workspace name
    "roboflow_project": "",    # Fill in your project name
    "roboflow_version": 1,
    
    # Data split
    "train_split": 0.8,
    "valid_split": 0.1,
    "test_split": 0.1,
    
    # Augmentation settings
    "augmentations": {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation": 15,  # Max degrees
        "brightness": 0.2,  # Adjustment range
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "blur": True,
        "noise": True,
    },
    
    # Class balancing
    "balance_classes": True,
    "min_samples_per_class": 10,
    "max_samples_per_class": 1000,
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "metrics": ["mAP", "mAP50", "mAP75", "precision", "recall"],
    "iou_thresholds": [0.5, 0.75, 0.9],
    "confidence_threshold": 0.5,
    "save_predictions": True,
    "save_visualizations": True,
    "visualizations_per_class": 5,  # Number of examples to visualize per class
}

# Deployment Configuration
DEPLOYMENT_CONFIG = {
    "export_format": "onnx",  # Options: "onnx", "torchscript", "coreml"
    "optimize": True,
    "quantize": False,  # INT8 quantization for edge devices
    "target_device": "gpu",  # Options: "gpu", "cpu", "edge"
}

# Logo-specific settings
LOGO_SETTINGS = {
    # Size thresholds (in pixels)
    "min_logo_size": 20,  # Filter out very small detections
    "max_logo_size": None,  # No upper limit by default
    
    # Aspect ratio constraints
    "min_aspect_ratio": 0.2,  # Width/Height
    "max_aspect_ratio": 5.0,
    
    # Confidence adjustments
    "small_logo_threshold": 0.6,  # Higher threshold for small logos
    "large_logo_threshold": 0.4,  # Lower threshold for large logos
    "size_threshold_cutoff": 50,  # Pixels (what counts as "small")
}

# Debug settings
DEBUG = {
    "visualize_predictions": True,
    "save_failed_detections": True,
    "log_level": "INFO",  # "DEBUG", "INFO", "WARNING", "ERROR"
    "profile_training": False,
    "check_dataset": True,  # Validate dataset before training
}

def get_config():
    """Return all configurations as a dictionary"""
    return {
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "inference": INFERENCE_CONFIG,
        "clip": CLIP_CONFIG,
        "dataset": DATASET_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "deployment": DEPLOYMENT_CONFIG,
        "logo_settings": LOGO_SETTINGS,
        "debug": DEBUG,
    }

def print_config():
    """Print current configuration"""
    config = get_config()
    print("=" * 60)
    print("RF-DETR Logo Detection Configuration")
    print("=" * 60)
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
