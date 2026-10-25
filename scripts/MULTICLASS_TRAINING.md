# OpenLogo + FlickrLogos32 multi-class training

The training pipeline uses OpenLogo and FlickrLogos32 as a brand-class dataset. `Logos_detection` stays separate because its only label is the generic class `Logo`.

## One-command run

From the project root, activate the project virtual environment and run:

```bash
source .venv/bin/activate
python scripts/prepare_train_multiclass.py --stage all --device 0 --epochs 100 --batch-size 8 --img-size 640 --workers 4
```

The script targets an 80/10/10 split, groups near-duplicate images, preserves source test images by default, audits the result, checks the class count, and then starts YOLOv7 training. The report records the actual split ratios. The best checkpoint is saved under `runs/train/openlogo_flickr_multiclass_v1/weights/best.pt`.

## Run stages separately

Prepare and audit only:

```bash
python scripts/prepare_train_multiclass.py --stage prepare
```

After fixing or confirming CUDA, train from the prepared dataset:

```bash
python scripts/prepare_train_multiclass.py --stage train --device 0 --epochs 100 --batch-size 8
```

`--stage train` re-runs the dataset audit before training. Set `--device cpu` only for a small smoke run; full multi-class training is intended for a CUDA GPU. Lower `--batch-size` if GPU memory is insufficient.

## Inputs and outputs

- `datasets/unification_inputs/openlogo.v1i.yolov7pytorch_data.yaml` and `flickrlogos32_data.yaml` provide portable, corrected paths while leaving downloaded source data unchanged.
- `scripts/logo_class_map_openlogo_flickr.yaml` merges clear text/figure/numbered variants into their brand. FlickrLogos32's `guiness` spelling maps to `guinness`.
- `datasets/logos_yolo_openlogo_flickr_multiclass/` contains `data.yaml`, split folders, a source manifest, and the unification report.
- `reports/dataset_audits/openlogo_flickr_multiclass/` contains the audit report, class/box CSVs, exact-duplicate report, and sample previews.

Use `--repartition-source-test` only when you intentionally want all source splits pooled and repartitioned. By default, source test images remain held out. Inspect `unification_report.json` and the audit before interpreting per-class test metrics; long-tail classes may still have too few independent examples or no examples in the source test split.

The source metadata records different licenses. Review attribution and redistribution requirements before publishing or sharing the unified images.
