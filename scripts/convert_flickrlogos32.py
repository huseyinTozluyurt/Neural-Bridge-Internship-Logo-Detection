"""
Convert FlickrLogos-32 dataset to COCO format
for use with RF-DETR training
"""

import json
import shutil
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import sys
import random
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_DIR

def parse_flickr_xml(xml_file):
    """Parse FlickrLogos-32 XML annotation file"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    annotations = []
    
    # Get image info
    size = root.find('size')
    if size is None or size.find('width') is None or size.find('height') is None:
        raise ValueError("annotation is missing image dimensions")
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Get all objects (logos)
    for obj in root.findall('object'):
        name_node = obj.find('name')
        name = name_node.text.strip() if name_node is not None and name_node.text else ""
        bndbox = obj.find('bndbox')
        if not name or bndbox is None:
            raise ValueError("object is missing its class name or bounding box")
        
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmax <= xmin or ymax <= ymin:
            raise ValueError(f"invalid bounding box {(xmin, ymin, xmax, ymax)} for {width}x{height} image")
        # Convert to COCO format [x, y, width, height]
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        area = (xmax - xmin) * (ymax - ymin)
        
        annotations.append({
            'class_name': name,
            'bbox': bbox,
            'area': area
        })
    
    if not annotations:
        raise ValueError("annotation contains no logo objects")
    return width, height, annotations

def convert_flickrlogos32(source_dir, dest_dir=None):
    """
    Convert FlickrLogos-32 dataset to COCO format
    
    Args:
        source_dir: Path to extracted FlickrLogos-32 folder
        dest_dir: Destination (default: datasets/logos/)
    """
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir) if dest_dir else DATASET_DIR
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"\n❌ Destination is not empty; refusing to overwrite existing data: {dest_path}")
        print("Choose a new output directory or move the existing output yourself.")
        return False
    
    print("\n" + "=" * 75)
    print("CONVERTING FLICKRLOGOS-32 TO COCO FORMAT")
    print("=" * 75)
    
    print(f"\nSource: {source_path}")
    print(f"Destination: {dest_path}")
    
    # Expected structure of FlickrLogos-32
    # FlickrLogos-32_dataset/
    #   classes/
    #     adidas/
    #     apple/
    #     ...
    #   FlickrLogos-32_trainvalset.txt
    #   FlickrLogos-32_testset.txt
    
    # Check if source exists
    if not source_path.exists():
        print(f"\n❌ Error: Source directory not found: {source_path}")
        print("\nPlease download FlickrLogos-32 first:")
        print("  1. https://www.kaggle.com/datasets/sujaykapadnis/flickrlogos32")
        print("  2. Extract the ZIP file")
        print("  3. Run this script with the path to the extracted folder")
        return False
    
    # Find the dataset folder
    dataset_folder = None
    for potential in [source_path, source_path / "FlickrLogos-32_dataset"]:
        if potential.exists() and (potential / "classes").exists():
            dataset_folder = potential
            break
    
    if not dataset_folder:
        print(f"\n❌ Error: Could not find 'classes' folder in {source_path}")
        print("\nExpected structure:")
        print("  FlickrLogos-32_dataset/")
        print("    classes/")
        print("      adidas/")
        print("      apple/")
        print("      ...")
        return False
    
    print(f"\n✓ Found dataset folder: {dataset_folder}")
    
    # Get all logo classes
    classes_dir = dataset_folder / "classes"
    logo_classes = sorted([d.name for d in classes_dir.iterdir() if d.is_dir()])
    
    print(f"✓ Found {len(logo_classes)} logo classes")
    
    # Create category mapping
    categories = [
        {"id": idx + 1, "name": class_name, "supercategory": "logo"}
        for idx, class_name in enumerate(logo_classes)
    ]
    
    class_to_id = {cat["name"]: cat["id"] for cat in categories}
    
    # Read train/test splits
    train_file = dataset_folder / "FlickrLogos-32_trainvalset.txt"
    test_file = dataset_folder / "FlickrLogos-32_testset.txt"
    
    if not train_file.exists() or not test_file.exists():
        print("\n⚠ Warning: Train/test split files not found")
        print("  Will create a deterministic 70/15/15 train/valid/test split")
        train_images = []
        test_images = []
        use_auto_split = True
    else:
        with open(train_file, encoding="utf-8") as f:
            train_images = [line.split()[0] for line in f if line.strip()]
        with open(test_file, encoding="utf-8") as f:
            test_images = [line.split()[0] for line in f if line.strip()]
        use_auto_split = False
        print(f"✓ Using provided splits: {len(train_images)} train, {len(test_images)} test")
    
    # Build deterministic, disjoint split membership once. Official trainval is
    # divided into train/valid per class; official test remains untouched.
    split_members = {"train": {}, "valid": {}, "test": {}}
    def normalize_split_entry(entry):
        normalized = entry.replace("\\", "/").strip("./")
        if normalized.startswith("classes/"):
            normalized = normalized[len("classes/"):]
        return normalized

    train_set = {normalize_split_entry(entry) for entry in train_images}
    test_set = {normalize_split_entry(entry) for entry in test_images}
    for class_name in logo_classes:
        image_files = sorted(list((classes_dir / class_name).glob("*.jpg")) + list((classes_dir / class_name).glob("*.png")))
        if use_auto_split:
            shuffled = list(image_files)
            random.Random(f"flickrlogos32:{class_name}:42").shuffle(shuffled)
            n = len(shuffled)
            train_end, valid_end = int(0.7 * n), int(0.85 * n)
            split_members["train"][class_name] = shuffled[:train_end]
            split_members["valid"][class_name] = shuffled[train_end:valid_end]
            split_members["test"][class_name] = shuffled[valid_end:]
        else:
            class_train = [f for f in image_files if f.name in train_set or f"{class_name}/{f.name}" in train_set]
            class_test = [f for f in image_files if f.name in test_set or f"{class_name}/{f.name}" in test_set]
            overlap = {f.name for f in class_train} & {f.name for f in class_test}
            if overlap:
                raise ValueError(f"FlickrLogos split files overlap for {class_name}: {sorted(overlap)[:3]}")
            shuffled = list(class_train)
            random.Random(f"flickrlogos32:{class_name}:42").shuffle(shuffled)
            valid_count = int(len(shuffled) * 0.1)
            split_members["train"][class_name] = shuffled[valid_count:]
            split_members["valid"][class_name] = shuffled[:valid_count]
            split_members["test"][class_name] = class_test

    # Process each split
    for split_name in ("train", "valid", "test"):
        print(f"\n{'=' * 75}")
        print(f"Processing {split_name.upper()} split...")
        print("=" * 75)
        
        split_dir = dest_path / split_name
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        coco_images = []
        coco_annotations = []
        image_id = 1
        annotation_id = 1
        
        skipped_images = 0
        # Process each logo class
        for class_name in logo_classes:
            split_images = split_members[split_name][class_name]
            
            # Process each image
            for img_file in split_images:
                dest_img = images_dir / f"{class_name}_{img_file.name}"
                
                # Get image dimensions
                try:
                    with Image.open(img_file) as img:
                        width, height = img.size
                except Exception:
                    print(f"  ⚠ Warning: Could not open {img_file.name}, skipping")
                    skipped_images += 1
                    continue

                xml_file = img_file.with_suffix('.xml')
                if not xml_file.is_file():
                    print(f"  ⚠ Warning: Missing annotation {xml_file.name}; skipping image")
                    skipped_images += 1
                    continue
                try:
                    ann_width, ann_height, annots = parse_flickr_xml(xml_file)
                    if (ann_width, ann_height) != (width, height):
                        raise ValueError(f"XML size {ann_width}x{ann_height} differs from image {width}x{height}")
                    if any(ann['class_name'] not in class_to_id for ann in annots):
                        raise ValueError("annotation references a class absent from classes/")
                except Exception as e:
                    print(f"  ⚠ Warning: Invalid {xml_file.name}: {e}; skipping image")
                    skipped_images += 1
                    continue

                shutil.copy2(img_file, dest_img)
                
                # Add image info
                coco_images.append({
                    "id": image_id,
                    "file_name": dest_img.name,
                    "width": width,
                    "height": height
                })
                
                for ann in annots:
                    coco_annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_to_id[ann['class_name']],
                        "bbox": ann['bbox'],
                        "area": ann['area'],
                        "iscrowd": 0
                    })
                    annotation_id += 1
                
                image_id += 1
        
        # Create COCO JSON
        coco_data = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories
        }
        
        # Save JSON
        json_file = split_dir / "_annotations.coco.json"
        with open(json_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"✓ Created {len(coco_images)} images")
        print(f"✓ Created {len(coco_annotations)} annotations")
        if skipped_images:
            print(f"⚠ Skipped {skipped_images} images with unreadable files or invalid/missing annotations")
        print(f"✓ Saved to {split_dir}")
    
    print("\n" + "=" * 75)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 75)
    print(f"\nDataset location: {dest_path}")
    print(f"Classes: {len(logo_classes)}")
    print(f"  {', '.join(logo_classes[:10])}{'...' if len(logo_classes) > 10 else ''}")
    print("\n✓ Dataset is ready for training!")
    print("\nNext steps:")
    print("  1. Verify: python scripts/download_dataset.py (option 4)")
    print("  2. Start training: Phase 3")
    print()
    
    return True

if __name__ == "__main__":
    print("\n" + "╔" + "=" * 73 + "╗")
    print("║" + " " * 20 + "FLICKRLOGOS-32 CONVERTER" + " " * 28 + "║")
    print("╚" + "=" * 73 + "╝")
    
    parser = argparse.ArgumentParser(description="Convert FlickrLogos-32 to COCO format")
    parser.add_argument("source_dir", help="Path to the extracted FlickrLogos-32 folder")
    parser.add_argument("--output-dir", help="Fresh output directory (defaults to datasets/logos)")
    args = parser.parse_args()
    convert_flickrlogos32(args.source_dir, args.output_dir)
