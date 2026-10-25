"""
Dataset download and preparation script for logo detection
Supports: Roboflow, LogoDet-3K, FlickrLogos-32
"""

import os
import sys
from pathlib import Path
from collections import Counter
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_DIR, ROBOFLOW_API_KEY, DATASET_CONFIG

def download_roboflow_dataset(workspace: str, project: str, version: int = 1):
    """
    Download dataset from Roboflow Universe
    
    Args:
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version number
    """
    from roboflow import Roboflow
    
    if not ROBOFLOW_API_KEY:
        print("❌ Error: ROBOFLOW_API_KEY not set!")
        print("Set it in .env file or export ROBOFLOW_API_KEY='your_key'")
        return False
    
    try:
        print(f"Downloading dataset from Roboflow...")
        print(f"  Workspace: {workspace}")
        print(f"  Project: {project}")
        print(f"  Version: {version}")
        
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("coco", location=str(DATASET_DIR))
        
        print(f"✓ Dataset downloaded successfully to {DATASET_DIR}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def download_flickrlogos32():
    """
    Download FlickrLogos-32 dataset
    Quick start option with 32 logo classes
    """
    print("=" * 60)
    print("FlickrLogos-32 Download Instructions")
    print("=" * 60)
    print("\nThis dataset contains 32 logo classes (5K+ images).")
    print("Popular logos: Nike, Adidas, Coca-Cola, Apple, etc.")
    print("\nDownload options:")
    print("1. Roboflow Universe: https://universe.roboflow.com/search?q=flickr-logos")
    print("2. Academic source: http://image.ntua.gr/iva/datasets/flickr_logos/")
    print("\nAfter download, organize as:")
    print(f"  {DATASET_DIR}/")
    print("    train/")
    print("      images/")
    print("      _annotations.coco.json")
    print("    valid/")
    print("      images/")
    print("      _annotations.coco.json")
    print("    test/")
    print("      images/")
    print("      _annotations.coco.json")
    print()

def download_logodet3k():
    """
    Download LogoDet-3K dataset
    Production-ready with 3000+ classes
    """
    print("=" * 60)
    print("LogoDet-3K Download Instructions")
    print("=" * 60)
    print("\nThis dataset contains 3,000+ logo classes (194K+ images).")
    print("Comprehensive logo detection dataset for production use.")
    print("\nDownload options:")
    print("1. GitHub: https://github.com/Wangjing1551/LogoDet-3K-Dataset")
    print("2. Roboflow Universe: https://universe.roboflow.com/search?q=logodet")
    print("\nDataset statistics:")
    print("  - 3,000+ logo classes")
    print("  - 194,000+ annotated instances")
    print("  - Real-world scenarios with occlusions, deformations")
    print("\nAfter download, organize as:")
    print(f"  {DATASET_DIR}/")
    print("    train/")
    print("      images/")
    print("      _annotations.coco.json")
    print("    valid/")
    print("      images/")
    print("      _annotations.coco.json")
    print("    test/")
    print("      images/")
    print("      _annotations.coco.json")
    print()

def check_dataset_structure():
    """
    Verify dataset structure and format
    """
    print("=" * 60)
    print("Dataset Structure Check")
    print("=" * 60)
    
    required_splits = ["train", "valid", "test"]
    all_good = True
    
    for split in required_splits:
        split_dir = DATASET_DIR / split
        images_dir = split_dir / "images"
        annotations_file = split_dir / "_annotations.coco.json"
        
        print(f"\nChecking {split} split...")
        
        # Check directory exists
        if not split_dir.exists():
            print(f"  ❌ Directory not found: {split_dir}")
            all_good = False
            continue
        else:
            print(f"  ✓ Directory exists: {split_dir}")
        
        # Check images directory
        if not images_dir.exists():
            print(f"  ❌ Images directory not found: {images_dir}")
            all_good = False
        else:
            image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
            print(f"  ✓ Images directory exists: {image_count} images")
        
        # Check annotations file
        if not annotations_file.exists():
            print(f"  ❌ Annotations file not found: {annotations_file}")
            all_good = False
        else:
            print(f"  ✓ Annotations file exists")
            
            # Load and check annotations
            try:
                with open(annotations_file, 'r') as f:
                    data = json.load(f)
                
                num_images = len(data.get('images', []))
                num_annotations = len(data.get('annotations', []))
                num_categories = len(data.get('categories', []))
                
                print(f"    - {num_images} images")
                print(f"    - {num_annotations} annotations")
                print(f"    - {num_categories} categories")
                
                # Check class distribution
                if num_annotations > 0:
                    category_ids = [ann['category_id'] for ann in data['annotations']]
                    class_counts = Counter(category_ids)
                    print(f"    - Class distribution:")
                    print(f"      Min: {min(class_counts.values())} samples")
                    print(f"      Max: {max(class_counts.values())} samples")
                    print(f"      Mean: {sum(class_counts.values()) / len(class_counts):.1f} samples")
                
            except Exception as e:
                print(f"  ❌ Error reading annotations: {e}")
                all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("✓ Dataset structure is valid!")
    else:
        print("❌ Dataset structure has issues. Please fix them before training.")
    print("=" * 60)
    
    return all_good

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("Logo Dataset Download & Preparation")
    print("=" * 60)
    print()
    
    print("Choose an option:")
    print("1. Download from Roboflow (requires API key)")
    print("2. FlickrLogos-32 instructions (quick start)")
    print("3. LogoDet-3K instructions (production)")
    print("4. Check existing dataset structure")
    print("5. Exit")
    print()
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == "1":
        workspace = input("Enter Roboflow workspace name: ").strip()
        project = input("Enter project name: ").strip()
        version = input("Enter version (default=1): ").strip() or "1"
        download_roboflow_dataset(workspace, project, int(version))
        
    elif choice == "2":
        download_flickrlogos32()
        
    elif choice == "3":
        download_logodet3k()
        
    elif choice == "4":
        check_dataset_structure()
        
    elif choice == "5":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice!")
        return
    
    # After download, offer to check structure
    if choice in ["1", "2", "3"]:
        check = input("\nWould you like to check the dataset structure? (y/n): ").strip().lower()
        if check == "y":
            check_dataset_structure()

if __name__ == "__main__":
    main()
