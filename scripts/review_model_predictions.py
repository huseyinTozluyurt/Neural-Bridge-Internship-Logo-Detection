#!/usr/bin/env python3
"""Run YOLOv7 on a review/unlabelled image set and create a human review queue.

Review ``review_queue.csv`` by filling the ``decision`` column with accept,
reject, or needs_correction. Outputs are drafts; this script never edits labels.
Do not use a held-out test split as a source for training-label review.
"""
from __future__ import annotations
import argparse, csv, subprocess, sys
from pathlib import Path
import cv2
import yaml

PROJECT=Path(__file__).resolve().parent.parent
EXTS={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights",required=True); ap.add_argument("--source",required=True)
    ap.add_argument("--output",type=Path,default=Path("runs/prediction_review")); ap.add_argument("--data",type=Path,help="Optional data.yaml for readable class names")
    ap.add_argument("--device",default="0"); ap.add_argument("--conf",type=float,default=.05); ap.add_argument("--img-size",type=int,default=640)
    a=ap.parse_args(); src=Path(a.source).expanduser().resolve(); weights=Path(a.weights).expanduser(); weights=weights if weights.is_absolute() else PROJECT/weights
    if not src.exists(): ap.error(f"Source not found: {src}")
    if not weights.is_file(): ap.error(f"Weights not found: {weights}")
    out=a.output.expanduser(); out=out if out.is_absolute() else PROJECT/out; out=out.resolve(); out.mkdir(parents=True,exist_ok=True)
    detect=PROJECT/"yolov7"/"detect.py"
    cmd=[sys.executable,str(detect),f"--weights={weights.resolve()}",f"--source={src}",f"--project={out}","--name=raw","--exist-ok","--save-txt","--save-conf",f"--conf-thres={a.conf}",f"--img-size={a.img_size}",f"--device={a.device}"]
    subprocess.run(cmd,cwd=PROJECT,check=True)
    labels={};
    if a.data:
        cfg=yaml.safe_load(a.data.expanduser().resolve().read_text(encoding="utf-8")); raw=cfg.get("names",[]); labels={int(k):str(v) for k,v in raw.items()} if isinstance(raw,dict) else dict(enumerate(map(str,raw)))
    source_files=[src] if src.is_file() and src.suffix.lower() in EXTS else sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXTS) if src.is_dir() else []
    pred_dir=out/"raw"/"labels"; rows=[]
    for image in source_files:
        # YOLO detect saves label files by image stem (nested sources are not always preserved).
        lp=pred_dir/f"{image.stem}.txt"
        if not lp.is_file(): continue
        im=cv2.imread(str(image)); h,w=im.shape[:2] if im is not None else (0,0)
        for idx,line in enumerate(lp.read_text(encoding="utf-8").splitlines()):
            v=line.split()
            if len(v)<5: continue
            try: cls=int(float(v[0])); x,y,bw,bh=map(float,v[1:5]); conf=float(v[5]) if len(v)>5 else ""
            except ValueError: continue
            rows.append({"image":str(image),"prediction_index":idx,"class_id":cls,"class_name":labels.get(cls,f"class_{cls}"),"confidence":conf,"x_center":x,"y_center":y,"width":bw,"height":bh,"decision":"pending","corrected_class":"","corrected_box_xywh":""})
    queue=out/"review_queue.csv"
    fields=["image","prediction_index","class_id","class_name","confidence","x_center","y_center","width","height","decision","corrected_class","corrected_box_xywh"]
    with queue.open("w",newline="",encoding="utf-8") as f: wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    (out/"REVIEW_README.txt").write_text("Review annotated images under raw/. In review_queue.csv, change decision to accept, reject, or needs_correction. Fill corrected_class and corrected_box_xywh as needed. This file is not imported into training automatically. Keep held-out test images out of training.\n",encoding="utf-8")
    print(f"Created {len(rows)} review rows: {queue}"); return 0
if __name__=="__main__": raise SystemExit(main())
