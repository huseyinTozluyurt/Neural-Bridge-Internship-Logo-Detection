"""
Enhanced dataset download helper with Universe dataset search
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_DIR, ROBOFLOW_API_KEY

def search_universe_datasets():
    """Search for popular logo datasets on Roboflow Universe"""
    
    print("\n" + "=" * 75)
    print("SEARCHING FOR LOGO DATASETS ON ROBOFLOW UNIVERSE")
    print("=" * 75)
    
    print("\n🔍 Popular Logo Detection Datasets:\n")
    
    datasets = [
        {
            "name": "Logo Detection - 3 Classes",
            "workspace": "roboflow-100",
            "project": "logo-detection-sfnyx",
            "version": 1,
            "classes": 3,
            "images": "~1K",
            "url": "https://universe.roboflow.com/roboflow-100/logo-detection-sfnyx"
        },
        {
            "name": "Brand Logo Detection",
            "workspace": "aiprojects-rnpqk",
            "project": "brand-logo-detection",
            "version": 1,
            "classes": 10,
            "images": "~500",
            "url": "https://universe.roboflow.com/aiprojects-rnpqk/brand-logo-detection"
        },
        {
            "name": "Company Logos",
            "workspace": "school-kgfzz",
            "project": "company-logos",
            "version": 1,
            "classes": 20,
            "images": "~800",
            "url": "https://universe.roboflow.com/school-kgfzz/company-logos"
        },
        {
            "name": "Logo Classification",
            "workspace": "logo-rgvtl",
            "project": "logo-l7gjg",
            "version": 1,
            "classes": 15,
            "images": "~600",
            "url": "https://universe.roboflow.com/logo-rgvtl/logo-l7gjg"
        }
    ]
    
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds['name']}")
        print(f"   Classes: {ds['classes']} | Images: {ds['images']}")
        print(f"   Workspace: {ds['workspace']}")
        print(f"   Project: {ds['project']}")
        print(f"   URL: {ds['url']}")
        print()
    
    print("=" * 75)
    print("\n💡 TIP: You can also search at https://universe.roboflow.com/search?q=logo")
    print("=" * 75)
    
    return datasets

def download_by_url():
    """Download dataset by pasting the Universe URL"""
    from roboflow import Roboflow
    
    print("\n" + "=" * 75)
    print("DOWNLOAD BY URL")
    print("=" * 75)
    
    print("\n1. Go to: https://universe.roboflow.com/search?q=logo")
    print("2. Find a dataset you like")
    print("3. Copy the full URL")
    print("4. Paste it below\n")
    
    url = input("Enter dataset URL (or 'back' to return): ").strip()
    
    if url.lower() == 'back':
        return False
    
    # Parse URL
    # Format: https://universe.roboflow.com/workspace/project/version
    try:
        parts = url.replace('https://', '').replace('http://', '').split('/')
        if 'universe.roboflow.com' in parts[0]:
            workspace = parts[1]
            project = parts[2]
            version = int(parts[3]) if len(parts) > 3 else 1
            
            print(f"\n✓ Parsed URL:")
            print(f"  Workspace: {workspace}")
            print(f"  Project: {project}")
            print(f"  Version: {version}")
            
            confirm = input("\nProceed with download? (y/n): ").strip().lower()
            if confirm == 'y':
                return download_dataset(workspace, project, version)
            
    except Exception as e:
        print(f"\n❌ Could not parse URL: {e}")
        print("\nExpected format: https://universe.roboflow.com/workspace/project/version")
    
    return False

def download_dataset(workspace: str, project: str, version: int = 1):
    """Download a dataset from Roboflow"""
    from roboflow import Roboflow
    
    if not ROBOFLOW_API_KEY:
        print("\n❌ Error: ROBOFLOW_API_KEY not set!")
        print("Please check your .env file")
        return False
    
    print("\n" + "=" * 75)
    print("DOWNLOADING DATASET")
    print("=" * 75)
    
    print(f"\n📥 Downloading from Roboflow Universe...")
    print(f"  Workspace: {workspace}")
    print(f"  Project: {project}")
    print(f"  Version: {version}")
    print(f"  Destination: {DATASET_DIR}")
    print()
    
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(
            "coco",
            location=str(DATASET_DIR.parent),  # Download to datasets/
            overwrite=True
        )
        
        print("\n" + "=" * 75)
        print("✅ DOWNLOAD SUCCESSFUL!")
        print("=" * 75)
        print(f"\nDataset location: {DATASET_DIR}")
        print("\n📊 Next steps:")
        print("  1. Verify dataset structure")
        print("  2. Check class distribution")
        print("  3. Proceed to training (Phase 3)")
        print()
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 75)
        print("❌ DOWNLOAD FAILED")
        print("=" * 75)
        print(f"\nError: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Check if the dataset exists at the URL")
        print("  2. Try a different dataset from Universe")
        print("  3. Verify your API key has access")
        print("  4. Some datasets may be private")
        print()
        return False

def main():
    print("\n" + "╔" + "=" * 73 + "╗")
    print("║" + " " * 20 + "LOGO DATASET DOWNLOADER" + " " * 30 + "║")
    print("╚" + "=" * 73 + "╝")
    
    while True:
        print("\n" + "=" * 75)
        print("MAIN MENU")
        print("=" * 75)
        print("\n1. Search popular logo datasets")
        print("2. Download by URL (paste Universe link)")
        print("3. Download from suggested list")
        print("4. Manual download instructions")
        print("5. Exit")
        print()
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            datasets = search_universe_datasets()
            print("\nWould you like to download one of these? (y/n): ", end="")
            if input().strip().lower() == 'y':
                try:
                    idx = int(input("\nEnter dataset number (1-4): ").strip()) - 1
                    if 0 <= idx < len(datasets):
                        ds = datasets[idx]
                        print(f"\n✓ Selected: {ds['name']}")
                        download_dataset(ds['workspace'], ds['project'], ds['version'])
                except (ValueError, IndexError):
                    print("Invalid selection")
        
        elif choice == "2":
            download_by_url()
        
        elif choice == "3":
            print("\n" + "=" * 75)
            print("SUGGESTED DATASETS")
            print("=" * 75)
            
            suggestions = [
                ("roboflow-100", "logo-detection-sfnyx", 1, "Logo Detection - 3 classes, ~1K images"),
                ("aiprojects-rnpqk", "brand-logo-detection", 1, "Brand Logos - 10 classes"),
                ("school-kgfzz", "company-logos", 1, "Company Logos - 20 classes"),
            ]
            
            for i, (ws, proj, ver, desc) in enumerate(suggestions, 1):
                print(f"\n{i}. {desc}")
                print(f"   Workspace: {ws}")
                print(f"   Project: {proj}")
            
            try:
                idx = int(input("\nSelect dataset (1-3) or 0 to cancel: ").strip())
                if 1 <= idx <= len(suggestions):
                    ws, proj, ver, desc = suggestions[idx - 1]
                    print(f"\n✓ Selected: {desc}")
                    download_dataset(ws, proj, ver)
            except (ValueError, IndexError):
                print("Invalid selection")
        
        elif choice == "4":
            print("\n" + "=" * 75)
            print("MANUAL DOWNLOAD INSTRUCTIONS")
            print("=" * 75)
            print("""
For LogoDet-3K (Production - 3000+ classes):
  1. Visit: https://github.com/Wangjing1551/LogoDet-3K-Dataset
  2. Download dataset (may require registration)
  3. Extract and organize in COCO format
  4. Place in: datasets/logos/

For FlickrLogos-32 (Quick Start - 32 classes):
  1. Visit: http://image.ntua.gr/iva/datasets/flickr_logos/
  2. Download dataset
  3. Convert to COCO format (if needed)
  4. Place in: datasets/logos/

Required structure:
  datasets/logos/
    ├── train/
    │   ├── images/
    │   └── _annotations.coco.json
    ├── valid/
    │   ├── images/
    │   └── _annotations.coco.json
    └── test/
        ├── images/
        └── _annotations.coco.json

See DATASET_GUIDE.md for more details.
            """)
            input("\nPress Enter to continue...")
        
        elif choice == "5":
            print("\nExiting...")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()
