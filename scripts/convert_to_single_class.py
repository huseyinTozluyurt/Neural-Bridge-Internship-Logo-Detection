"""Explicitly collapse a multi-class dataset into one generic ``logo`` class.

This discards brand identity. The command requires explicit confirmation and a
fresh output directory so stale files cannot silently accumulate.
"""

import argparse
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_DIR

def convert_to_single_class(source_path=None, yolo_dataset_path=None):
    """Convert all YOLO labels to single class (0 = logo)"""

    yolo_dataset_path = Path(yolo_dataset_path) if yolo_dataset_path else DATASET_DIR.parent / "logos_yolo_single_class"
    source_path = Path(source_path) if source_path else DATASET_DIR.parent / "logos_yolo"
    yolo_dataset_path = yolo_dataset_path.expanduser().resolve()
    source_path = source_path.expanduser().resolve()
    if not (source_path / "data.yaml").is_file():
        raise FileNotFoundError(f"Source data.yaml not found: {source_path / 'data.yaml'}")
    if yolo_dataset_path.exists() and any(yolo_dataset_path.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {yolo_dataset_path}. "
            "Choose a new output path; stale files are never reused."
        )

    print("\n" + "=" * 75)
    print("CONVERTING TO SINGLE-CLASS LOGO DETECTION")
    print("=" * 75)
    print(f"\nSource: {source_path}")
    print(f"Destination: {yolo_dataset_path}")
    print()

    # Create directories
    for split in ["train", "valid", "test"]:
        (yolo_dataset_path / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dataset_path / split / "labels").mkdir(parents=True, exist_ok=True)

    total_converted = 0
    total_annotations = 0

    # Process each split
    for split in ["train", "valid", "test"]:
        print(f"Processing {split} split...")

        source_images = source_path / split / "images"
        source_labels = source_path / split / "labels"

        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"Expected {split} image and label folders under {source_path}")
        image_stems = {
            path.stem for path in source_images.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

        dest_images = yolo_dataset_path / split / "images"
        dest_labels = yolo_dataset_path / split / "labels"

        # Copy images so the generated dataset remains portable when moved.
        num_images = 0
        for img_file in source_images.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                dest_img = dest_images / img_file.name
                shutil.copy2(img_file, dest_img)
                num_images += 1

        # Convert labels to class 0
        num_labels = 0
        num_annots = 0
        for label_file in source_labels.glob("*.txt"):
            if label_file.stem not in image_stems:
                continue
            dest_label = dest_labels / label_file.name

            with open(label_file, 'r') as f_in:
                lines = f_in.readlines()

            with open(dest_label, 'w') as f_out:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Change class ID to 0, keep bbox coordinates
                        parts[0] = '0'
                        f_out.write(' '.join(parts) + '\n')
                        num_annots += 1

            num_labels += 1

        print(f"  ✓ {num_images} images copied")
        print(f"  ✓ {num_labels} label files converted")
        print(f"  ✓ {num_annots} annotations → class 0 (logo)")

        total_converted += num_labels
        total_annotations += num_annots

    # Create single-class data.yaml
    data_yaml = yolo_dataset_path / "data.yaml"
    with open(data_yaml, 'w') as f:
        f.write(f"# Logo Detection Dataset - Single Class YOLO Format\n\n")
        f.write("path: .\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['logo']\n")

    print()
    print("=" * 75)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 75)
    print(f"\nSingle-class dataset: {yolo_dataset_path}")
    print(f"Data config: {data_yaml}")
    print(f"Total labels converted: {total_converted}")
    print(f"Total annotations: {total_annotations}")
    print()
    print("�� Dataset structure:")
    for split in ["train", "valid", "test"]:
        num_images = len(list((yolo_dataset_path / split / "images").glob("*")))
        num_labels = len(list((yolo_dataset_path / split / "labels").glob("*.txt")))
        print(f"  • {split}: {num_images} images, {num_labels} labels")
    print()
    print("🎯 Next step:")
    print(f"  Train generic logo detection: python scripts/train_yolo.py --data {data_yaml} --expect-classes 1")
    print()

    return yolo_dataset_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATASET_DIR.parent / "logos_yolo", help="Source multi-class YOLO dataset directory")
    parser.add_argument("--output", type=Path, default=DATASET_DIR.parent / "logos_yolo_single_class", help="New output directory")
    parser.add_argument("--confirm-single-class", action="store_true", help="Confirm that brand IDs will be discarded")
    args = parser.parse_args()
    if not args.confirm_single_class:
        parser.error("This conversion discards brand identities. Pass --confirm-single-class only when generic logo detection is intended.")
    convert_to_single_class(args.source, args.output)
