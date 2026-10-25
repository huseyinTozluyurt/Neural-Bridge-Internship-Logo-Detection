#!/usr/bin/env python3
"""Build contact sheets for manual review of YOLO labels in each split."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import yaml

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=Path("reports/dataset_review"))
    ap.add_argument("--per-split", type=int, default=100)
    ap.add_argument("--columns", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.data.expanduser().resolve().read_text(encoding="utf-8"))
    ypath = a.data.expanduser().resolve(); root = Path(cfg.get("path", ypath.parent)); root = root if root.is_absolute() else ypath.parent / root
    names = cfg.get("names", []); names = [str(names[k]) for k in sorted(names, key=lambda x:int(x))] if isinstance(names, dict) else list(names)
    out = a.output.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    import random
    rng = random.Random(a.seed)
    for split in ("train", "val", "test"):
        if split not in cfg: continue
        values = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
        files=[]
        for v in values:
            d=Path(v); d=d if d.is_absolute() else root/d
            files.extend(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
        files=sorted(set(files)); rng.shuffle(files); files=files[:max(0,a.per_split)]
        tiles=[]
        for p in files:
            im=cv2.imread(str(p))
            if im is None: continue
            h,w=im.shape[:2]; images_root=next((parent for parent in p.parents if parent.name.lower()=="images"),None)
            label=(images_root.parent/"labels"/p.relative_to(images_root).with_suffix(".txt")) if images_root else (p.parent/"labels"/p.with_suffix(".txt").name)
            if label.is_file():
                for line in label.read_text(encoding="utf-8",errors="replace").splitlines():
                    z=line.split()
                    if len(z)!=5: continue
                    try: c=int(z[0]); x,y,bw,bh=map(float,z[1:])
                    except ValueError: continue
                    x1=max(0,int((x-bw/2)*w)); y1=max(0,int((y-bh/2)*h)); x2=min(w-1,int((x+bw/2)*w)); y2=min(h-1,int((y+bh/2)*h))
                    cv2.rectangle(im,(x1,y1),(x2,y2),(60,220,90),2)
                    cv2.putText(im,names[c] if 0<=c<len(names) else f"class_{c}",(x1,max(16,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(60,220,90),1,cv2.LINE_AA)
            else: cv2.putText(im,"NO LABEL FILE",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
            tile_w=320; scale=tile_w/w; im=cv2.resize(im,(tile_w,max(1,int(h*scale))))
            cv2.putText(im,str(p.name)[:45],(6,im.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            tiles.append(im)
        if not tiles: print(f"{split}: no readable images"); continue
        cell_w=320; cell_h=max(t.shape[0] for t in tiles); cols=max(1,a.columns); rows=(len(tiles)+cols-1)//cols
        sheet=__import__("numpy").full((rows*cell_h,cols*cell_w,3),35,dtype="uint8")
        for i,t in enumerate(tiles):
            x=(i%cols)*cell_w; y=(i//cols)*cell_h; sheet[y:y+t.shape[0],x:x+cell_w]=t
        target=out/f"{split}_contact_sheet.jpg"; cv2.imwrite(str(target),sheet); print(f"{split}: {target} ({len(tiles)} images)")
    return 0
if __name__=="__main__": raise SystemExit(main())
