#!/usr/bin/env python3
"""Run a training/evaluation command and record reproducibility metadata."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

PROJECT=Path(__file__).resolve().parent.parent
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--output",type=Path,default=Path("reports/experiments")); ap.add_argument("--data",type=Path); ap.add_argument("--weights",type=Path); ap.add_argument("--name",default=None); ap.add_argument("command",nargs=argparse.REMAINDER,help="Command to run, after --"); a=ap.parse_args()
    cmd=a.command[1:] if a.command and a.command[0]=="--" else a.command
    if not cmd: ap.error("Provide a command after --")
    out=a.output.expanduser(); out=out if out.is_absolute() else PROJECT/out; out=out.resolve(); out.mkdir(parents=True,exist_ok=True)
    def resolved(p):
        if p is None:return None
        v=p.expanduser(); return (v if v.is_absolute() else PROJECT/v).resolve()
    data=resolved(a.data); weights=resolved(a.weights)
    def file_info(p): return {"path":str(p),"sha256":sha(p)} if p and p.is_file() else ({"path":str(p),"exists":False} if p else None)
    try: git=subprocess.run(["git","-C",str(PROJECT),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    except Exception: git=None
    start=datetime.now(timezone.utc).isoformat(); print("Running:"," ".join(cmd),flush=True); result=subprocess.run(cmd,cwd=PROJECT); end=datetime.now(timezone.utc).isoformat()
    try:
        import torch; cuda=torch.cuda.is_available(); device=torch.cuda.get_device_name(0) if cuda else "cpu"
    except Exception: cuda=False; device="unknown"
    record={"name":a.name or Path(cmd[0]).name,"command":cmd,"started_utc":start,"finished_utc":end,"return_code":result.returncode,"project_root":str(PROJECT),"git_commit":git,"python":sys.version,"platform":platform.platform(),"cuda_available":cuda,"gpu":device,"dataset":file_info(data),"checkpoint":file_info(weights),"working_directory":os.getcwd()}
    target=out/f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{(a.name or 'run').replace(' ','_')}.json"; target.write_text(json.dumps(record,indent=2),encoding="utf-8"); print(f"Run metadata: {target}"); return result.returncode
if __name__=="__main__": raise SystemExit(main())
