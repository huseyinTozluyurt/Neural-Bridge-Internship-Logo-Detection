#!/usr/bin/env python3
"""Calibrate detector and CLIP thresholds from a labeled CSV calibration set.

CSV columns: detector_confidence, clip_similarity, truth where truth is one of
known, unknown_logo, non_logo. The chosen thresholds optimize macro-F1; report
the per-class scores and use a separate untouched test set for final claims.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

CLASSES=("known","unknown_logo","non_logo")
def macro_f1(truth,pred):
    scores=[]
    for c in CLASSES:
        tp=sum(t==c and p==c for t,p in zip(truth,pred)); fp=sum(t!=c and p==c for t,p in zip(truth,pred)); fn=sum(t==c and p!=c for t,p in zip(truth,pred))
        scores.append(2*tp/max(1,2*tp+fp+fn))
    return sum(scores)/len(scores),dict(zip(CLASSES,scores))
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--input",required=True,type=Path); ap.add_argument("--output",type=Path,default=Path("reports/open_set_calibration.json")); ap.add_argument("--steps",type=int,default=41); a=ap.parse_args()
    rows=[]
    with a.input.expanduser().resolve().open(newline="",encoding="utf-8") as f:
        for n,r in enumerate(csv.DictReader(f),2):
            try: conf=float(r["detector_confidence"]); sim=float(r["clip_similarity"]); truth=r["truth"].strip().lower()
            except (KeyError,ValueError) as e: ap.error(f"Bad/missing values at CSV row {n}: {e}")
            if truth not in CLASSES: ap.error(f"truth must be one of {CLASSES} (row {n})")
            rows.append((conf,sim,truth))
    if not rows or any(not any(r[2]==c for r in rows) for c in CLASSES): ap.error("Calibration set must include examples of all three truth classes")
    steps=max(3,a.steps); best=None
    for i in range(steps):
        dt=i/(steps-1)
        for j in range(steps):
            st=-1+2*j/(steps-1)
            pred=["non_logo" if conf<dt else "known" if sim>=st else "unknown_logo" for conf,sim,_ in rows]
            score,per=macro_f1([r[2] for r in rows],pred)
            if best is None or score>best[0]: best=(score,dt,st,per)
    result={"source_csv":str(a.input.expanduser().resolve()),"examples":len(rows),"classes":{c:sum(r[2]==c for r in rows) for c in CLASSES},"objective":"macro_f1","thresholds":{"detector_confidence_min":best[1],"clip_similarity_min_for_known":best[2]},"macro_f1":best[0],"per_class_f1":best[3],"warning":"Thresholds are fitted to this calibration CSV. Do not report these as unbiased test metrics."}
    out=a.output.expanduser(); out=out if out.is_absolute() else Path.cwd()/out; out=out.resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(f"Best macro-F1 {best[0]:.3f}; detector confidence >= {best[1]:.3f}, CLIP similarity >= {best[2]:.3f} for known. Saved {out}"); return 0
if __name__=="__main__": raise SystemExit(main())
