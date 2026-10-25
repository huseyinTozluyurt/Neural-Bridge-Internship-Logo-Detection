#!/usr/bin/env python3
"""Compare YOLOv7 checkpoints on the same video set and report wall-clock speed."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parent.parent
VIDEO={".mp4",".mov",".avi",".mkv",".m4v",".webm"}
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--weights",nargs="+",required=True); ap.add_argument("--source",required=True); ap.add_argument("--output",type=Path,default=Path("reports/video_benchmark")); ap.add_argument("--device",default="0"); ap.add_argument("--conf",type=float,default=.25); ap.add_argument("--img-size",type=int,default=640); a=ap.parse_args()
    src=Path(a.source).expanduser(); src=src if src.is_absolute() else PROJECT/src; src=src.resolve()
    videos=[src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO) if src.is_dir() else []
    if not videos: ap.error(f"No video files found under {src}")
    weights=[]
    for w in a.weights:
        p=Path(w).expanduser(); p=p if p.is_absolute() else PROJECT/p
        if not p.is_file(): ap.error(f"Checkpoint missing: {p}")
        weights.append(p.resolve())
    out=a.output.expanduser(); out=out if out.is_absolute() else PROJECT/out; out=out.resolve(); out.mkdir(parents=True,exist_ok=True); rows=[]
    for wi,w in enumerate(weights):
        for vi,v in enumerate(videos):
            name=f"model{wi:02d}_video{vi:03d}"; cmd=[sys.executable,str(PROJECT/"yolov7"/"detect.py"),f"--weights={w}",f"--source={v}",f"--project={out/'runs'}",f"--name={name}","--exist-ok",f"--device={a.device}",f"--conf-thres={a.conf}",f"--img-size={a.img_size}"]
            start=time.perf_counter(); proc=subprocess.run(cmd,cwd=PROJECT,check=False); elapsed=time.perf_counter()-start
            cap=__import__("cv2").VideoCapture(str(v)); frames=cap.get(__import__("cv2").CAP_PROP_FRAME_COUNT); fps=cap.get(__import__("cv2").CAP_PROP_FPS); cap.release()
            rows.append({"checkpoint":str(w),"video":str(v),"return_code":proc.returncode,"wall_seconds":round(elapsed,3),"video_frames":int(frames),"video_fps":fps,"processing_fps":round(frames/elapsed,3) if elapsed>0 else 0,"output_dir":str(out/"runs"/name)})
            print(f"{w.name} | {v.name}: {elapsed:.1f}s, {frames/elapsed if elapsed else 0:.1f} processing fps, exit={proc.returncode}")
    with (out/"benchmark.csv").open("w",newline="",encoding="utf-8") as f: wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    (out/"benchmark.json").write_text(json.dumps({"device":a.device,"confidence":a.conf,"image_size":a.img_size,"results":rows,"note":"Wall-clock includes model load and output writing. Compare identical hardware/settings; this is a speed benchmark, not accuracy evaluation."},indent=2),encoding="utf-8"); return 0 if all(r["return_code"]==0 for r in rows) else 1
if __name__=="__main__": raise SystemExit(main())
