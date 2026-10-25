#!/usr/bin/env python3
"""Evaluate tracker boxes against a hand-labeled CSV.

Both CSVs use frame,track_id,x1,y1,x2,y2 columns. Use the same frame numbering
and coordinate system. Reports detection precision/recall, ID switches,
fragmentations, and a conservative IDF1 approximation from box matches.
"""
from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path

def read(path):
    result=defaultdict(list)
    with Path(path).expanduser().resolve().open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try: result[int(row["frame"])].append({"id":str(row["track_id"]),"box":tuple(float(row[k]) for k in ("x1","y1","x2","y2"))})
            except (KeyError,ValueError) as e: raise ValueError(f"Invalid tracking CSV row {row}: {e}")
    return result
def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); area=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/area if area else 0
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--ground-truth",required=True); ap.add_argument("--predictions",required=True); ap.add_argument("--output",type=Path,default=Path("reports/tracking_evaluation.json")); ap.add_argument("--iou",type=float,default=.5); a=ap.parse_args()
    gt=read(a.ground_truth); pr=read(a.predictions); tp=fp=fn=0; pair_counts=Counter(); last_match={}; was_matched={}; switches=fragments=0
    for frame in sorted(set(gt)|set(pr)):
        candidates=sorted(((iou(g["box"],p["box"]),gi,pi) for gi,g in enumerate(gt[frame]) for pi,p in enumerate(pr[frame]) if iou(g["box"],p["box"])>=a.iou),reverse=True)
        used_g=set(); used_p=set(); matches=[]
        for score,gi,pi in candidates:
            if gi in used_g or pi in used_p: continue
            used_g.add(gi); used_p.add(pi); matches.append((gt[frame][gi]["id"],pr[frame][pi]["id"]))
        tp+=len(matches); fp+=len(pr[frame])-len(used_p); fn+=len(gt[frame])-len(used_g)
        current=set()
        for gid,pid in matches:
            current.add(gid); pair_counts[(gid,pid)]+=1
            if gid in last_match and last_match[gid]!=pid: switches+=1
            if was_matched.get(gid) is False: fragments+=1
            was_matched[gid]=True; last_match[gid]=pid
        for obj in gt[frame]:
            if obj["id"] not in current and obj["id"] in was_matched: was_matched[obj["id"]]=False
    # Greedily pair persistent identities by their matched-frame counts.
    used_gt=set(); used_pred=set(); idtp=0
    for (g,p),n in sorted(pair_counts.items(),key=lambda x:x[1],reverse=True):
        if g not in used_gt and p not in used_pred: used_gt.add(g); used_pred.add(p); idtp+=n
    total_gt=sum(map(len,gt.values())); total_pr=sum(map(len,pr.values())); idfp=total_pr-idtp; idfn=total_gt-idtp
    result={"ground_truth":str(Path(a.ground_truth).resolve()),"predictions":str(Path(a.predictions).resolve()),"iou_threshold":a.iou,"true_positives":tp,"false_positives":fp,"false_negatives":fn,"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"detection_f1":2*tp/max(1,2*tp+fp+fn),"id_switches":switches,"fragmentations":fragments,"idtp_approx":idtp,"idfp_approx":idfp,"idfn_approx":idfn,"idf1_approx":2*idtp/max(1,2*idtp+idfp+idfn),"note":"IDF1 uses greedy global identity pairing and should be treated as an approximation; use a standard MOT evaluator for benchmark reporting."}
    out=a.output.expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
