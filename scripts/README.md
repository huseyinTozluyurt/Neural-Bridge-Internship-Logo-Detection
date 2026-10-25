# Logo Detection Training & Inference Scripts

Unified scripts for YOLOv7 logo detection training and inference.

---

## Quick Start

### 1. Train a Model
```bash
python scripts/train_yolo.py \
    --data datasets/openlogo_yolo/data.yaml \
    --epochs 100 \
    --batch-size 16
```

For brand-specific FlickrLogos-32 detection, train the 32-class dataset and guard against accidentally selecting a generic single-class dataset:
```bash
python scripts/train_yolo.py \
    --data datasets/logos_yolo_multiclass_ready/data.yaml \
    --expect-classes 32 \
    --min-val-objects-per-class 5 \
    --weights yolov7/yolov7.pt \
    --epochs 100 --img-size 640 --batch-size 4 \
    --name flickrlogos32_multiclass_v1
```
Adjust batch size to fit available GPU memory. This command will stop until every brand has at least five validation objects; fix the validation split first and keep the test split untouched. Do not warm-start from a `logo_single_*` checkpoint because it has a one-class output head. `--single-cls` deliberately removes brand identity and should stay off for this task.

### 2. Run Detection
```bash
python scripts/detect_yolo.py \
    --weights runs/train/openlogo_training3/weights/best.pt \
    --source your_images/ \
    --conf 0.4
```

### 3. Image and Video Detection GUI
The PyQt5 desktop interface lets you switch between `test_images/` and `test_videos/`, check one or more files or folders, choose the checkpoint and detection settings, and follow progress and logs without entering the options in the terminal. The window sizes itself to the available screen, and its split panes can be resized by dragging. You can also drag local image/video files or folders from your file manager into the media tree. When detection completes, select a result to view its annotated image or play its annotated video in the same window. Install the optional GUI dependency and launch it:
```bash
pip install -r scripts/requirements-gui.txt
python scripts/video_detection_gui.py
```

Checking a folder selects every supported image or video recursively beneath it, based on the selected dataset. Each selected media file gets its own YOLO output folder; a timestamped batch folder also contains `summary.csv` and `summary.json`. The GUI can stop the current item and mark remaining selections as cancelled. It currently processes local files; a YouTube or other online source can be added later.

The command-line picker is still available for terminal workflows:
```bash
python scripts/batch_video_detect.py \
    --weights runs/train/logo_single_microsoft_v3/weights/best.pt \
    --device 0
```

The picker starts at `test_videos/`. Use `--root another/folder` to choose a different starting folder, or `--device cpu` when CUDA is unavailable. Detection currently runs once per video, so the checkpoint is reloaded for each selected file.

### 4. Detect Known and Unknown Logos
`detect_unknown_logos.py` uses a YOLOv7 checkpoint to propose logo regions, then compares each region with CLIP reference-image prototypes. A candidate is labeled with the closest known brand when its cosine similarity reaches the threshold; otherwise it is labeled `unknown`.

Install OpenAI CLIP into the active project environment (in addition to the YOLOv7 requirements):
```bash
pip install git+https://github.com/openai/CLIP.git
```

Prepare a gallery with one folder per known brand, containing clear example logo images:
```text
known_logos/
├── adidas/
│   ├── example_1.jpg
│   └── example_2.png
└── nike/
    ├── example_1.jpg
    └── example_2.png
```

Run it on an image or image directory:
```bash
python scripts/detect_unknown_logos.py \
    --weights runs/train/logo_single_microsoft_v3/weights/best.pt \
    --source test_images/ \
    --known-logos known_logos/ \
    --device 0 \
    --similarity-threshold 0.78
```

Results are written to `runs/unknown_logo_detection/`: annotated images, crops grouped by matched brand or `unknown`, `detections.csv`, and `summary.json`. Lower the similarity threshold to accept more brand matches; raise it to classify more candidates as unknown. The detector must itself propose unseen logos to make them eligible for matching, so a detector trained as one generic `logo` class is preferable to one trained only on a fixed set of brands. The similarity threshold should be calibrated on representative known and unknown logo crops; CLIP matching is a useful baseline, not a guarantee that every unknown logo or non-logo will be classified correctly.

### 5. Audit a YOLO Dataset
Create class and box statistics, check for exact duplicate images across splits, and save annotated previews:
```bash
python scripts/audit_dataset.py \
    --data datasets/logos_yolo_single_class/data.yaml \
    --output reports/dataset_audit
```

The output directory contains `audit_report.json`, CSV summaries, exact duplicate listings, and up to 20 annotated previews per split. Use `--preview-count 0` to skip previews or `--skip-hashes` to skip exact duplicate detection. The audit reports findings; it does not modify the dataset.

### 6. Evaluate Saved Checkpoints on the Test Split
Evaluate one or more YOLOv7 checkpoints without retraining:
```bash
python scripts/evaluate_yolo.py \
    --data datasets/logos_yolo_single_class/data.yaml \
    --weights runs/train/logo_single_microsoft_v3/weights/best.pt \
    --device 0
```

Pass multiple checkpoint paths after `--weights` to compare runs. The evaluator checks the checkpoint's embedded class count and name order against `data.yaml` before scoring, then runs YOLOv7's test routine and saves per-class AP50/AP50-95, precision/recall at the global best-F1 point, fixed-threshold precision/recall/F1, false-positive and missed-logo CSVs, and annotated examples. If an older checkpoint has no embedded class names, use `--allow-unverified-class-mapping` only after confirming its original class order manually. Defaults are confidence 0.25 for the error lists and IoU 0.5 for matching; tune with `--report-conf` and `--match-iou`. Outputs are written under `reports/checkpoint_evaluation/`.

### 7. Prepare Leakage-Aware Splits and Review Labels

Create a new dataset from configured train/val/test image directories. Exact byte-identical files are grouped into one split, source files stay untouched, and the result includes `data.yaml` plus a seed/assignment manifest:
```bash
python scripts/prepare_dataset_splits.py --data datasets/logos_yolo/data.yaml --output datasets/logos_yolo_multiclass_ready --ratios 0.8 0.1 0.1 --seed 42 --preserve-test
```
`--preserve-test` keeps the current held-out test images fixed and repartitions only train/validation; it aborts if it finds exact duplicates crossing into test. This groups byte-identical images but does not find near-duplicates or adjacent video frames. For video-derived data, group by source video before training. Review a random sample of existing annotations with:
```bash
python scripts/review_dataset_samples.py --data datasets/logos_yolo_resplit/data.yaml --output reports/dataset_review
```
The script creates one labeled contact sheet per split. It does not modify labels.

### 8. Unify Multiple Logo Datasets

Combine multiple YOLO object-detection datasets, including converted OpenLogo, FlickrLogos, and Roboflow exports, into one canonical class list and train/validation/test splits. Source datasets are not modified. The tool remaps class IDs by class name, accepts Roboflow's `valid` split name as well as `val`, groups visually near-duplicate images so they stay in one split, and writes a manifest and class-support report. It requires every image to have a YOLO label file by default; an empty file means a confirmed negative image.

Start with the example alias map and edit it for the datasets you are combining:
```yaml
# scripts/logo_class_map.example.yaml
classes: auto
aliases:
  Coca-Cola: [coca cola, coca-cola, coke]
  Guinness: [guiness]
  Adidas: [adidas logo]
  Apple: [apple inc.]
  Starbucks: [starbucks coffee]
```

With `classes: auto`, the script uses the union of all source classes. It normalizes case, spaces, and punctuation, but it will not guess that semantic variants such as `Coke` and `Coca-Cola` are the same brand; list those in `aliases`. If you set an explicit `classes` list to train a selected taxonomy, the default unmapped-label behavior is an error. You may pass `--unmapped-labels exclude-image` to omit an entire image containing an unmapped annotation; it never drops only that box, which would create an incompletely labeled training image. Use that option only when the source annotations cover all logos in each image. Missing label files are an error unless you pass `--allow-missing-labels-as-negative` for a source that intentionally omits labels on confirmed negatives. The unifier cannot infer brands from a generic `logo` annotation: do not mix generic-logo labels into a brand classifier unless you intentionally define and annotate a compatible generic/unknown class.

For a Pascal VOC/OpenLogo source, first preserve its annotated brand names in YOLO format:
```bash
python scripts/convert_voc_to_yolo.py \
    --source /path/to/openlogo \
    --output datasets/openlogo_yolo_multiclass \
    --seed 42
```

FlickrLogos-32 is converted to COCO by the project's existing converter. Convert that COCO export to YOLO before including it:
```bash
python scripts/convert_flickrlogos32.py /path/to/FlickrLogos-32 --output-dir datasets/flickrlogos_coco
python scripts/convert_coco_to_yolo.py --input datasets/flickrlogos_coco --output datasets/flickrlogos_yolo
```
Roboflow versions can be downloaded directly in YOLOv7 format. The COCO converter now builds one category-name mapping across all splits and rejects malformed or out-of-bounds boxes, so split-local COCO category IDs cannot silently change brand meanings.

Then pass the prepared `data.yaml` from each source:
```bash
python scripts/unify_logo_datasets.py \
    --data datasets/logos_yolo_multiclass_ready/data.yaml \
           datasets/openlogo_yolo_multiclass/data.yaml \
           datasets/logos_detection_yolo/data.yaml \
           datasets/flickrlogos_yolo/data.yaml \
    --class-map scripts/logo_class_map.example.yaml \
    --output datasets/logos_unified_v1 \
    --ratios 0.8 0.1 0.1 \
    --seed 42
```

The output contains `data.yaml`, YOLO `train/`, `val/`, and `test/` image/label folders, `unification_manifest.json`, and `unification_report.json`. Existing source test images are kept in the output test split by default, and duplicate groups touching a source test image stay together in test. The requested ratios apply to the images available for train/validation; the report records both requested and actual split ratios. The script stops if any class has no training examples or if validation has no positive boxes; it records missing validation/test class coverage as warnings. Use `--repartition-source-test` only when you deliberately want to reassign source test images. Previously inspected source tests are not a new independent benchmark.

The default perceptual hash distance is 4 bits; use `--duplicate-hamming 0` for byte-identical grouping only. Perceptual grouping is a safeguard, not proof of no leakage—review the report's duplicate groups and manually group images from the same original source or video. Use `--min-images-per-class N` after reviewing support to require at least N independent duplicate groups for every class in each requested split; broad datasets with hundreds of rare classes may not meet that guard. Output filenames and split assignment are stable for the same ordered input list, class map, source contents, ratios, and seed.

Run the project audit on the generated dataset before training:
```bash
python scripts/audit_dataset.py \
    --data datasets/logos_unified_v1/data.yaml \
    --output reports/logos_unified_v1_audit
```

Then train with the generated YAML. Set `--expect-classes` to the `nc` value in the generated file if you want the training guard:
```bash
python scripts/train_yolo.py \
    --data datasets/logos_unified_v1/data.yaml \
    --expect-classes <nc-from-data-yaml> \
    --weights yolov7/yolov7.pt \
    --epochs 100 --img-size 640 --batch-size 4 \
    --name logos_unified_v1
```

The output class IDs are newly assigned from the canonical class list and are not compatible with old checkpoints unless class names and order match exactly. YOLO detection is closed-set: brands outside its class list are not automatically identified as unknown.

For an unlabeled or review-only image folder, generate prediction overlays and a CSV queue:
```bash
python scripts/review_model_predictions.py --weights runs/train/logo_single_microsoft_v3/weights/best.pt --source review_images --data datasets/logos_yolo_single_class/data.yaml
```
Review `runs/prediction_review/review_queue.csv` and annotated files under `raw/`. Predictions are not imported automatically. Keep the final held-out test set out of training-label review.

### 9. Calibrate Unknown-Logo Thresholds

Prepare a CSV with `detector_confidence,clip_similarity,truth` columns. Use `known`, `unknown_logo`, or `non_logo` labels from a calibration set separate from the final test set:
```bash
python scripts/calibrate_unknown_logos.py --input reports/open_set_calibration_examples.csv --output reports/open_set_calibration.json
```
The script selects detector and CLIP thresholds by macro-F1 and saves per-class results. It does not automatically change `detect_unknown_logos.py`; apply thresholds explicitly and validate them on an untouched test set.

### 10. Benchmark Video Runtime and Tracking

Compare checkpoints on the same videos and hardware:
```bash
python scripts/benchmark_video_detection.py --weights runs/train/model_a/weights/best.pt runs/train/model_b/weights/best.pt --source test_videos --device 0
```
The report includes processing FPS and wall time, including model loading and output writing. It measures speed, not accuracy.

For tracking evaluation, prepare ground-truth and prediction CSVs with `frame,track_id,x1,y1,x2,y2` columns and run:
```bash
python scripts/evaluate_tracking.py --ground-truth tracking_ground_truth.csv --predictions tracking_predictions.csv --iou 0.5
```
The report includes detection precision/recall, ID switches, fragmentations, and an explicitly approximate IDF1. Use a standard MOT evaluator for benchmark-grade identity metrics.

### 11. Record Reproducible Runs

Wrap an existing training or evaluation command to record its command, environment, code revision, and optional dataset/checkpoint hashes:
```bash
python scripts/record_experiment.py --name logo_training --data datasets/logos_yolo/data.yaml -- python scripts/train_yolo.py --data datasets/logos_yolo/data.yaml --epochs 100
```
Run records are saved under `reports/experiments/`. The wrapper records metadata and exit code; it does not change training settings.

### 12. Run the Logo Reflection Pipeline

Detects logos with YOLO + optional EasyOCR, segments with GrabCut or MobileSAM, flips horizontally (with perspective correction if needed), blends back into image.

**Example Command:**
```bash
python .\scripts\logo_flipping_pipeline.py \
    --yolo-weights runs/train/openlogo_training3/weights/best.pt \
    --input ..\reflected_logos\nike5.jpg \
    --output ..\reflected_logos\nike5_out.jpg \
    --artifact-dir ..\reflected_logos\artifacts \
    --naturalise-mode line \
    --blend-mode alpha \
    --use-easyocr \
    --save-intermediates \
    --log-level DEBUG
```

**Parameters Used:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--yolo-weights` | Path to .pt | Trained YOLOv7 model weights |
| `--input` | Path to image | Input image with reflected logo |
| `--output` | Path to image | Output image with flipped logo |
| `--artifact-dir` | Path to folder | Directory for intermediate files |
| `--naturalise-mode` | line | Flipping method (line/polygon/auto) |
| `--blend-mode` | alpha | Blending method (alpha/poisson/feather) |
| `--use-easyocr` | Flag | Enable EasyOCR text detection |
| `--save-intermediates` | Flag | Save masks, overlays to artifact-dir |
| `--log-level` | DEBUG | Logging verbosity |

**Additional Available Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yolo-conf` | 0.4 | YOLO confidence threshold |
| `--yolo-iou` | 0.45 | YOLO IOU threshold for NMS |
| `--yolo-img-size` | 1280 | YOLO inference image size |
| `--disable-ocr` | Disabled | Disable EasyOCR (opposite of --use-easyocr) |
| `--easyocr-conf` | 0.1 | EasyOCR confidence threshold |
| `--easyocr-langs` | en | EasyOCR language codes |
| `--merge-gap` | 12 | Gap for merging overlapping detections |
| `--grabcut-iters` | 5 | GrabCut iterations for mask refinement |
| `--min-mask-quality` | 0.05 | Minimum mask coverage ratio (5%) |
| `--use-sam-masker` | Disabled | Use MobileSAM for refined masks |
| `--sam-model` | mobile_sam.pt | Path to MobileSAM weights |
| `--use-perspective-flip` | True | Enable perspective-aware flipping |
| `--no-perspective-flip` | Flag | Disable perspective flipping |
| `--inpaint-expansion` | 5 | Mask dilation for inpainting context |
| `--inpaint-mask-margin` | 0 | Mask dilation for logo replacement |

**Output Files:**
Intermediates saved to `artifact-dir/<image_name>/`:
- `masked_logos/`: Extracted logo regions
- `masks/`: Binary segmentation masks
- `overlays/`: Visualization overlays

### 13. Video Logo Tracking & Flipping

**Tracking:**
YOLO detects logos per-frame with confidence filtering. CLIP embeddings enable re-identification when tracks are lost. Optical flow (Lucas-Kanade) tracks keypoints between frames. Fine-grid tracking divides logos into cells for robust motion estimation with outlier rejection. Kalman filter predicts position during detection gaps with velocity decay.

**Segmentation:**
MobileSAM segments logos using multi-point prompting (center + corners). GrabCut post-refinement cleans mask boundaries. Masks are cached for occlusion handling.

**Flipping:**
PCA estimates the logo's symmetry line with angle clamping (±45° from vertical). Affine transform reflects the logo across this axis. Alpha blending with valid pixel masking excludes warp artifacts.

**Inpainting:**
LaMa (ML-based) or OpenCV Telea fills the logo region with plausible background. Inpainted frames are cached for occlusion compositing.

**Occlusion Handling:**
MediaPipe segments hands in each frame. Occlusion is detected when hand masks overlap logo regions (coverage analysis). Unoccluded logo patches (< 20% occlusion) are cached with pre-computed flip. During occlusion, the cached logo is pasted at original size (no scaling) centered on Kalman-predicted position, over the cached inpainted background.

**Example Command:**
```bash
python .\scripts\video_logo_tracking.py \
    --input ..\reflected_videos\IMG_4331.MOV \
    --output-viz ..\reflected_videos\IMG_4331\ann.mp4 \
    --output-clean ..\reflected_videos\IMG_4331\fli.mp4 \
    --yolo-conf 0.5 \
    --yolo-weights .\runs\train\openlogo_training3\weights\best.pt \
    --disable-ocr \
    --input-is-flipped \
    --use-sam-masker \
    --use-fine-grid-tracking \
    --log-level DEBUG \
    --inpaint-radius 7 \
    --inpaint-expansion 0 \
    --inpaint-mask-margin 22 \
    --use-lama-inpainting
```

**Parameters Used:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--input` | Path to video | Input video file |
| `--output-viz` | Path to output | Visualization with tracking overlays, bboxes, IDs |
| `--output-clean` | Path to output | Clean video with flipped logos (no overlays) |
| `--yolo-conf` | 0.5 | Confidence threshold for YOLO detections |
| `--yolo-weights` | Path to .pt | Trained YOLOv7 model weights |
| `--disable-ocr` | Flag | Disable EasyOCR text detection (YOLO only) |
| `--input-is-flipped` | Flag | Input is flipped - unflip for detection, then remap |
| `--use-sam-masker` | Flag | Use MobileSAM for refined logo masks |
| `--use-fine-grid-tracking` | Flag | Enable per-cell optical flow tracking |
| `--log-level` | DEBUG | Logging verbosity (DEBUG/INFO/WARNING/ERROR) |
| `--inpaint-radius` | 10 | Telea inpainting radius in pixels |
| `--inpaint-expansion` | 20 | Mask dilation for inpainting context |
| `--inpaint-mask-margin` | 22 | Mask dilation for logo replacement |
| `--use-lama-inpainting` | Flag | Use LaMa inpainting instead of OpenCV Telea |

**Additional Available Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--yolo-iou` | 0.45 | IOU threshold for NMS |
| `--yolo-img-size` | 1280 | YOLO inference image size |
| `--device` | 0 | CUDA device (0, 1, 2) or 'cpu' |
| `--use-easyocr` | Disabled | Enable EasyOCR (opposite of --disable-ocr) |
| `--easyocr-conf` | 0.1 | EasyOCR confidence threshold |
| `--easyocr-langs` | en | EasyOCR language codes |
| `--merge-gap` | 12 | Gap for merging overlapping detections |
| `--grabcut-iters` | 5 | GrabCut iterations for mask refinement |
| `--min-mask-quality` | 0.05 | Minimum mask coverage ratio (5%) |
| `--sam-model` | mobile_sam.pt | Path to MobileSAM weights |
| `--sam-device` | Auto | Device for SAM (cuda:0 or cpu) |
| `--sam-imgsz` | 512 | MobileSAM predictor resolution |
| `--use-perspective-flip` | True | Enable perspective-aware logo flipping |
| `--no-perspective-flip` | Flag | Disable perspective flipping (simple flip) |
| `--pose-dir` | None | Directory with bodypose JSON files |
| `--no-pose-prediction` | Flag | Disable pose-based motion prediction |
| `--disable-clip` | Flag | Disable CLIP embeddings for tracking |
| `--clip-model` | ViT-B/32 | CLIP model for logo embeddings |
| `--clip-device` | Auto | Device for CLIP |
| `--disable-optical-flow` | Flag | Disable optical flow tracking |
| `--grid-cell-size` | 10 | Grid cell size for fine-grained tracking |

**Output Files:**
- `ann.mp4`: Visualization with bounding boxes, track IDs, confidence scores
- `fli.mp4`: Clean video with logos flipped to correct orientation
- `inp.mp4`: Background with logos inpainted (auto-generated)

---

## Training Script (`train_yolo.py`)

Unified wrapper for training YOLOv7 models on logo datasets.

### Basic Usage

```bash
# Standard training on OpenLogo dataset
python scripts/train_yolo.py --data datasets/openlogo_yolo/data.yaml --epochs 100
```

### Advanced Options

```bash
# Custom batch size and image size
python scripts/train_yolo.py \
    --data datasets/openlogo_yolo/data.yaml \
    --epochs 100 \
    --batch-size 32 \
    --img-size 800 \
    --name my_custom_training

# Resume from checkpoint
python scripts/train_yolo.py \
    --data datasets/openlogo_yolo/data.yaml \
    --resume

# Quick test run (5 epochs, small batch)
python scripts/train_yolo.py \
    --data datasets/openlogo_yolo/data.yaml \
    --epochs 5 \
    --batch-size 8 \
    --name test_run

# Single-class detection (all logos as one class)
python scripts/train_yolo.py \
    --data datasets/openlogo_yolo/data.yaml \
    --single-cls

# CPU training (not recommended)
python scripts/train_yolo.py \
    --data datasets/openlogo_yolo/data.yaml \
    --device cpu
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to data.yaml file |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 16 | Training batch size |
| `--img-size` | 640 | Input image size (must be multiple of 32) |
| `--weights` | yolov7/yolov7.pt | Pre-trained weights path |
| `--cfg` | yolov7/cfg/training/yolov7.yaml | Model config file |
| `--hyp` | yolov7/data/hyp.scratch.p5.yaml | Hyperparameters file |
| `--project` | runs/train | Project directory |
| `--name` | logo_training | Run name |
| `--device` | 0 | CUDA device (0, 1, 2) or 'cpu' |
| `--workers` | 4 | Data loading workers |
| `--single-cls` | False | Train as single-class |
| `--exist-ok` | False | Allow existing project/name |
| `--resume` | False | Resume from last checkpoint |

### Output Files

Training creates the following structure:
```
runs/train/<name>/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   ├── last.pt          # Last epoch checkpoint
│   └── epoch_*.pt       # Intermediate checkpoints
├── results.txt          # Training metrics per epoch
├── results.png          # Training curves visualization
└── train_batch*.jpg     # Sample training batches
```

### Training Tips

**Batch Size:**
- RTX 4090 (24GB): 16-32
- RTX 3090 (24GB): 16-24
- RTX 3080 (10GB): 8-12
- RTX 3060 (12GB): 8-16

**Image Size:**
- 640: Standard, good balance
- 800: Better for small logos, slower training
- 384: Faster training, lower accuracy

**For Best Results:**
- Use `--single-cls` if detecting generic logos (not brand-specific)
- Train for at least 50-100 epochs
- Monitor results.txt for mAP@0.5 convergence
- Best model saved when validation mAP@0.5 improves

---

## Detection Script (`detect_yolo.py`)

Unified wrapper for running inference with trained YOLOv7 models.

### Basic Usage

```bash
# Detect on single image
python scripts/detect_yolo.py \
    --weights runs/train/openlogo_training3/weights/best.pt \
    --source logo_test/IMG_9480.JPG

# Detect on entire folder
python scripts/detect_yolo.py \
    --weights runs/train/openlogo_training3/weights/best.pt \
    --source logo_test/
```

### Advanced Options

```bash
# Lower confidence threshold (more detections, may have false positives)
python scripts/detect_yolo.py \
    --weights runs/train/openlogo_training3/weights/best.pt \
    --source logo_test/ \
    --conf 0.25

# Higher confidence threshold (fewer but more accurate detections)
python scripts/detect_yolo.py \
    --weights runs/train/openlogo_training3/weights/best.pt \
    --source logo_test/ \
    --conf 0.6

# Save labels only (no annotated images)
python scripts/detect_yolo.py \
    --weights runs/train/openlogo_training3/weights/best.pt \
    --source logo_test/ \
    --nosave

# Custom output location
python scripts/detect_yolo.py \
    --weights runs/train/openlogo_training3/weights/best.pt \
    --source logo_test/ \
    --project my_results \
    --name experiment1

# Process video
python scripts/detect_yolo.py \
    --weights runs/train/openlogo_training3/weights/best.pt \
    --source video.mp4 \
    --conf 0.4

# Webcam detection
python scripts/detect_yolo.py \
    --weights runs/train/openlogo_training3/weights/best.pt \
    --source 0 \
    --view-img
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--weights` | Required | Path to model weights (.pt file) |
| `--source` | Required | Input (image/folder/video/webcam) |
| `--conf` | 0.4 | Confidence threshold (0.0-1.0) |
| `--iou` | 0.45 | IOU threshold for NMS |
| `--img-size` | 640 | Inference image size |
| `--project` | runs/detect | Output directory |
| `--name` | detection | Run name |
| `--device` | 0 | CUDA device or 'cpu' |
| `--no-save-txt` | False | Don't save label files |
| `--no-save-conf` | False | Don't save confidences |
| `--nosave` | False | Don't save images/videos |
| `--classes` | None | Filter by class (e.g., --classes 0 2 3) |
| `--view-img` | False | Display results |
| `--exist-ok` | False | Allow existing output folder |

### Output Files

Detection creates the following structure:
```
runs/detect/<name>/
├── *.jpg               # Annotated images with bounding boxes
└── labels/
    └── *.txt           # YOLO format detection labels
```

Label format (one detection per line):
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

### Confidence Threshold Guide

| Threshold | Use Case | Trade-off |
|-----------|----------|-----------|
| 0.25-0.35 | Maximum recall, find all logos | More false positives |
| 0.35-0.45 | **Balanced (recommended)** | Good precision/recall balance |
| 0.45-0.60 | High precision, critical applications | May miss some logos |
| 0.60+ | Very high confidence only | Will miss many logos |

**Recommendation:** Start with 0.4, adjust based on results:
- Too many false positives? Increase to 0.5-0.6
- Missing logos? Decrease to 0.3-0.35

## Dataset Management Scripts

### download_dataset.py
Interactive dataset downloader for Roboflow, LogoDet-3K, and FlickrLogos-32.

```bash
python scripts/download_dataset.py
```

### convert_voc_to_yolo.py
Convert Pascal VOC format to multi-class YOLO format, preserving each VOC `object/name` as its class. The conversion creates reproducible, approximately class-balanced train/validation/test splits and a `data.yaml` ready for the unifier. It refuses to overwrite non-empty output and writes a conversion report.

```bash
python scripts/convert_voc_to_yolo.py --source /path/to/openlogo
```

By default, the source is `<project>/openlogo` and output is
`<project>/datasets/openlogo_yolo`. Use `--source` and `--output` to choose
other locations. Relative paths are resolved from the project directory, so
the script can be launched from another current directory:

```bash
python scripts/convert_voc_to_yolo.py \
    --source openlogo \
    --output datasets/openlogo_yolo_multiclass
```

The converter preserves the actual class names found in the VOC annotations; older generated outputs that have `nc: 1` and `names: ['logo']` have already lost brand identities and should be regenerated from the original VOC files into a new empty output directory before multi-class training.

### convert_coco_to_yolo.py
Convert COCO format to YOLO format.

```bash
python scripts/convert_coco_to_yolo.py --input datasets/logos --output datasets/logos_yolo
```

The converter requires `train` and `valid`/`val` COCO splits, accepts an optional `test` split, remaps category IDs consistently by category name across splits, and refuses to overwrite a non-empty output directory.

### convert_to_single_class.py
Convert multi-class YOLO dataset to single-class.

```bash
python scripts/convert_to_single_class.py \
    --source datasets/logos_yolo \
    --output datasets/logos_yolo_single_class_new \
    --confirm-single-class
```
This conversion discards brand labels and now refuses to write into a non-empty destination.

### convert_flickrlogos32.py
Convert FlickrLogos-32 to COCO format.

```bash
python scripts/convert_flickrlogos32.py /path/to/FlickrLogos-32
```

### generate_sample_dataset.py
Generate synthetic sample dataset for testing.

```bash
python scripts/generate_sample_dataset.py
```

---

## Model Performance

**Current Best Model:** `runs/train/openlogo_training3/weights/best.pt`

| Metric | Value |
|--------|-------|
| mAP@0.5 | 81.1% |
| mAP@0.5:0.95 | 59.7% |
| Precision | 80.5% |
| Recall | 73.7% |
| Dataset | OpenLogo (27,083 images) |
| Training Time | ~6 hours @ 100 epochs (RTX 4090) |

**Detection Performance on Test Set:**
- Confidence 0.4: 6/7 images detected (85.7%)
- Confidence 0.35: 7/7 images detected (100%)
- Confidence 0.5: 6/7 images detected (85.7%)

---

## File Locations

### Code
- **YOLOv7 implementation**: `yolov7/`
- **Training scripts**: `scripts/train_yolo.py`
- **Detection scripts**: `scripts/detect_yolo.py`
- **Dataset utilities**: `scripts/download_dataset.py`, `scripts/convert_*.py`

### Weights
- **Pre-trained YOLOv7**: `yolov7/yolov7.pt` (73 MB)
- **Best trained model**: `runs/train/openlogo_training3/weights/best.pt` (285 MB)
- **Training checkpoints**: `runs/train/*/weights/`

### Datasets
- **OpenLogo (YOLO format)**: `datasets/openlogo_yolo/` (27,083 images)
- **Archived datasets**: `archive/datasets_old/`

### Results
- **Training runs**: `runs/train/`
- **Detection results**: `runs/detect/`
- **Archived results**: `archive/training_runs_old/`

---

## Advanced Features

### Perspective-Aware Logo Flipping

The pipeline intelligently detects and handles logos with perspective distortion (e.g., on curved surfaces, angled views, clothing).

**How It Works:**
1. **Detection**: Checks if logo polygon has >15% difference in opposite edge lengths
2. **Unwarp**: Transforms logo to canonical rectangular view
3. **Flip**: Flips horizontally in canonical view
4. **Rewarp**: Transforms back to original perspective

**Use Cases:**
- Logos on curved surfaces (bottles, cans)
- Perspective shots with visible distortion  
- Logos on fabric and draped clothing
- Angled camera views

**Controls:**
```bash
# Enable (default)
python scripts/logo_flipping_pipeline.py --use-perspective-flip --input logo.jpg --output flipped.jpg

# Disable for simple flip (faster)
python scripts/logo_flipping_pipeline.py --no-perspective-flip --input logo.jpg --output flipped.jpg
```

**Modes:**
- `--naturalise-mode auto`: Auto-choose between line/polygon-based flipping
- `--naturalise-mode polygon`: Always use polygon-based approach
- `--naturalise-mode line`: Use symmetry line-based reflection

### Flip-Detect-Unflip Strategy

Both the pipeline and video tracker implement an intelligent detection strategy for mirrored/flipped input:

**Problem**: YOLO and OCR models are trained on normal (non-mirrored) images. When you detect on already-flipped content:
- Text appears reversed (OCR fails)
- Logos appear mirrored (lower YOLO confidence)
- Directional features don't match training data

**Solution**: The flip-detect-unflip strategy:

**For Normal Video Input:**
```
Normal Frame (from video)
    ↓
Detect on normal frame (YOLO + OCR) ← Good accuracy!
    ↓
Use bboxes as-is
    ↓
Track & process on normal frame
    ↓
Output: Visualization + Clean flipped video
```

**For Testing with Simulated Flipped Input (--input-is-flipped):**
```
Normal Frame (from video)
    ↓
Flip frame horizontally (simulate flipped video)
    ↓
Detect on ORIGINAL unflipped frame (YOLO + OCR) ← Good accuracy!
    ↓
Flip bboxes to match flipped frame: x_new = width - (x + w)
    ↓
Track & visualize on flipped frame
    ↓
Flip pipeline result to match orientation
    ↓
Output: Flipped visualization + Flipped clean video
```

**Benefits:**
- 20-40% improvement in detection accuracy on mirrored content
- OCR can actually read reversed text
- YOLO features align with training data
- No retraining required

**Implementation Details:**

For video tracking, the bbox flip transformation is:
```python
# Horizontal flip: x-coordinate mirrors across image center
x_flipped = image_width - (x + width)
# y, width, height remain unchanged
```

**Usage:**
```bash
# Normal video (default)
python scripts/video_logo_tracking.py --input video.mp4 --output-viz viz.mp4 --output-clean clean.mp4

# Testing mode: simulate flipped input
python scripts/video_logo_tracking.py --input video.mp4 --output-viz viz.mp4 --output-clean clean.mp4 --input-is-flipped
```

---

## Troubleshooting

### Out of Memory (OOM) Errors
```bash
# Reduce batch size
python scripts/train_yolo.py --data datasets/openlogo_yolo/data.yaml --batch-size 8

# Reduce image size
python scripts/train_yolo.py --data datasets/openlogo_yolo/data.yaml --img-size 384
```

### Cache File Errors (PyTorch 2.6+)
```bash
# Remove old cache files
rm datasets/openlogo_yolo/train/labels.cache
rm datasets/openlogo_yolo/valid/labels.cache

# Restart training - cache will regenerate
python scripts/train_yolo.py --data datasets/openlogo_yolo/data.yaml
```

### Missing Dependencies
```bash
pip install seaborn matplotlib pandas tensorboard
```

### Low Detection Rate
- Lower confidence threshold: `--conf 0.3`
- Check if logos are similar to training data
- Consider retraining with more similar examples

### Too Many False Positives
- Raise confidence threshold: `--conf 0.5` or `--conf 0.6`
- Check training data quality
- Consider adding negative samples to training set
