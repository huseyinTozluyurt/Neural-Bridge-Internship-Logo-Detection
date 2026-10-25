"""
Test script to verify RF-DETR and CLIP installation
"""
import torch
import clip
from PIL import Image
import numpy as np

def test_gpu():
    """Test GPU availability"""
    print("=" * 60)
    print("GPU Test")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("✗ CUDA is not available - will use CPU")
    print()

def test_rfdetr():
    """Test RF-DETR installation and basic inference"""
    print("=" * 60)
    print("RF-DETR Test")
    print("=" * 60)
    try:
        from rfdetr import RFDETRMedium
        print("✓ RF-DETR imported successfully")
        
        # Initialize model
        model = RFDETRMedium()
        print("✓ RF-DETR Medium model initialized")
        print(f"  Model loaded with pre-trained weights")
        
        # Create a dummy image for testing
        dummy_image = Image.new('RGB', (640, 480), color='white')
        print("✓ Created test image (640x480)")
        
        # Test prediction
        detections = model.predict(dummy_image, threshold=0.5)
        print(f"✓ Prediction successful - {len(detections)} detections on dummy image")
        print(f"  Model ready for fine-tuning or inference")
        
    except Exception as e:
        print(f"✗ RF-DETR test failed: {e}")
    print()

def test_clip():
    """Test CLIP installation"""
    print("=" * 60)
    print("CLIP Test")
    print("=" * 60)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✓ CLIP model loaded successfully")
        print(f"  Device: {device}")
        print(f"  Model: ViT-B/32")
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        
        # Test text prompts
        text = clip.tokenize(["a Nike logo", "an Adidas logo", "a Coca-Cola logo"]).to(device)
        
        # Test inference
        with torch.no_grad():
            image_input = preprocess(dummy_image).unsqueeze(0).to(device)
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text)
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        print(f"✓ CLIP inference successful")
        print(f"  Text prompts tested: 3")
        print(f"  Similarity scores computed successfully")
        
    except Exception as e:
        print(f"✗ CLIP test failed: {e}")
    print()

def test_other_dependencies():
    """Test other dependencies"""
    print("=" * 60)
    print("Other Dependencies Test")
    print("=" * 60)
    try:
        import supervision as sv
        print("✓ supervision imported successfully")
        
        from roboflow import Roboflow
        print("✓ roboflow imported successfully")
        
        import tensorboard
        print("✓ tensorboard imported successfully")
        
        import wandb
        print("✓ wandb imported successfully")
        
    except Exception as e:
        print(f"✗ Dependency test failed: {e}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RF-DETR Logo Detection - Setup Verification")
    print("=" * 60)
    print()
    
    test_gpu()
    test_rfdetr()
    test_clip()
    test_other_dependencies()
    
    print("=" * 60)
    print("Setup verification complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set up Roboflow account at https://roboflow.com")
    print("2. Set up W&B account (optional) at https://wandb.ai")
    print("3. Download logo dataset (LogoDet-3K or FlickrLogos-32)")
    print()
