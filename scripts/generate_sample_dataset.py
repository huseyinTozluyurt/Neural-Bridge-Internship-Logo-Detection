"""
Generate a sample logo dataset for testing the RF-DETR pipeline
This creates synthetic data to test the system before getting real logos
"""

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_DIR

# Logo classes (simulated brands)
LOGO_CLASSES = [
    {"id": 1, "name": "circle_logo", "supercategory": "logo"},
    {"id": 2, "name": "square_logo", "supercategory": "logo"},
    {"id": 3, "name": "triangle_logo", "supercategory": "logo"},
]

def generate_logo_image(width=640, height=480, num_logos=1):
    """Generate a synthetic image with logo-like shapes"""
    # Create image with random background
    bg_color = tuple(random.randint(200, 255) for _ in range(3))
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    annotations = []
    
    for _ in range(num_logos):
        # Random logo type
        logo_class = random.choice(LOGO_CLASSES)
        
        # Random position and size
        logo_size = random.randint(50, 150)
        x = random.randint(10, width - logo_size - 10)
        y = random.randint(10, height - logo_size - 10)
        
        # Random color (different from background)
        logo_color = tuple(random.randint(0, 150) for _ in range(3))
        
        # Draw logo based on class
        if logo_class["name"] == "circle_logo":
            draw.ellipse([x, y, x + logo_size, y + logo_size], fill=logo_color, outline=(0, 0, 0), width=3)
        elif logo_class["name"] == "square_logo":
            draw.rectangle([x, y, x + logo_size, y + logo_size], fill=logo_color, outline=(0, 0, 0), width=3)
        elif logo_class["name"] == "triangle_logo":
            points = [(x + logo_size//2, y), (x, y + logo_size), (x + logo_size, y + logo_size)]
            draw.polygon(points, fill=logo_color, outline=(0, 0, 0), width=3)
        
        # Create annotation
        annotations.append({
            "class_id": logo_class["id"],
            "bbox": [x, y, logo_size, logo_size],
            "area": logo_size * logo_size
        })
    
    return img, annotations

def create_coco_json(images_info, annotations_list):
    """Create COCO format JSON"""
    coco_data = {
        "images": images_info,
        "annotations": annotations_list,
        "categories": LOGO_CLASSES
    }
    return coco_data

def generate_dataset(num_train=50, num_valid=15, num_test=15):
    """Generate complete dataset with train/valid/test splits"""
    
    print("\n" + "=" * 75)
    print("GENERATING SAMPLE LOGO DATASET")
    print("=" * 75)
    
    print(f"\nGenerating:")
    print(f"  Train: {num_train} images")
    print(f"  Valid: {num_valid} images")
    print(f"  Test: {num_test} images")
    print()
    
    for split, num_images in [("train", num_train), ("valid", num_valid), ("test", num_test)]:
        print(f"Creating {split} split...")
        
        # Create directories
        split_dir = DATASET_DIR / split
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        images_info = []
        annotations_list = []
        annotation_id = 1
        
        for img_id in range(1, num_images + 1):
            # Generate image
            num_logos = random.randint(1, 3)  # 1-3 logos per image
            img, annotations = generate_logo_image(num_logos=num_logos)
            
            # Save image
            img_filename = f"logo_{split}_{img_id:04d}.jpg"
            img_path = images_dir / img_filename
            img.save(img_path)
            
            # Add image info
            images_info.append({
                "id": img_id,
                "file_name": img_filename,
                "width": img.width,
                "height": img.height
            })
            
            # Add annotations
            for ann in annotations:
                annotations_list.append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": ann["class_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0
                })
                annotation_id += 1
        
        # Save COCO JSON
        coco_data = create_coco_json(images_info, annotations_list)
        json_path = split_dir / "_annotations.coco.json"
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"  ✓ Created {num_images} images")
        print(f"  ✓ Created {len(annotations_list)} annotations")
        print(f"  ✓ Saved to {split_dir}")
    
    print("\n" + "=" * 75)
    print("✅ SAMPLE DATASET CREATED SUCCESSFULLY")
    print("=" * 75)
    print(f"\nDataset location: {DATASET_DIR}")
    print(f"\nDataset structure:")
    print(f"  • {num_train} training images")
    print(f"  • {num_valid} validation images")
    print(f"  • {num_test} test images")
    print(f"  • 3 logo classes (circle, square, triangle)")
    print()
    print("📊 Class distribution:")
    for cls in LOGO_CLASSES:
        print(f"  • {cls['name']}: Random distribution across images")
    print()
    print("🎯 Next steps:")
    print("  1. Verify dataset: python scripts/download_dataset.py (option 4)")
    print("  2. Start training: Phase 3")
    print("  3. Replace with real logo data later")
    print()
    print("⚠️  NOTE: This is synthetic data for testing only.")
    print("   For production, use real logo datasets (FlickrLogos-32, LogoDet-3K)")
    print("=" * 75)

if __name__ == "__main__":
    print("\n" + "╔" + "=" * 73 + "╗")
    print("║" + " " * 20 + "SAMPLE DATASET GENERATOR" + " " * 28 + "║")
    print("╚" + "=" * 73 + "╝")
    
    print("\nThis will create a synthetic logo dataset for testing.")
    print("The dataset contains simple geometric shapes as 'logos'.")
    print()
    print("⚠️  This is NOT real logo data - it's for testing the pipeline.")
    print()
    
    confirm = input("Generate sample dataset? (y/n): ").strip().lower()
    
    if confirm == 'y':
        # Ask for size
        print("\nDataset size options:")
        print("  1. Small (50 train, 15 valid, 15 test) - Quick test")
        print("  2. Medium (200 train, 50 valid, 50 test) - Better test")
        print("  3. Large (500 train, 100 valid, 100 test) - Full test")
        
        size = input("\nSelect size (1-3, default=1): ").strip() or "1"
        
        if size == "1":
            generate_dataset(50, 15, 15)
        elif size == "2":
            generate_dataset(200, 50, 50)
        elif size == "3":
            generate_dataset(500, 100, 100)
        else:
            print("Invalid choice, using small dataset")
            generate_dataset(50, 15, 15)
    else:
        print("\nDataset generation cancelled.")
        print("\nAlternative options:")
        print("  1. Download FlickrLogos-32 manually")
        print("  2. Download LogoDet-3K manually")
        print("  3. Create your own dataset in Roboflow")
        print("\nSee DATASET_GUIDE.md for instructions")
