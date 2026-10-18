"""Read Kaggle diagnostic JSON files without disclosing private raw logs."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_to_kaggle import load_credentials, SLUG
env, username = load_credentials()
kernel = f"{username}/{SLUG}"
allow = {"status","phase","mode","error","cpu_ram_gib","available_ram_gib","scratch_free_gib",
         "expected_download_gib","free_scratch_gib","download_missing_gib",
         "required_free_with_reserve_gib","gpu_count","backend","storage_profile",
         "success","gpu_strategy","model_total_gib","backend_assignment"}
def sanitize(value):
    s = str(value)
    s = re.sub(r"https?://\S+", "[URL]",s)
    s = re.sub(r"[\w.+-]+@[\w.-]+\.\w+", "[EMAIL]",s)
    s = re.sub(r"(?i)(token|key|secret|password|authorization)\s*[:=]\s*\S+",
               "[REDACTED SECRET]",s)
    s = re.sub(r"[A-Za-z0-9_+/=-]{50,}", "[REDACTED]",s)
    return s[:400]

with tempfile.TemporaryDirectory() as d:
    command = [sys.executable,"-m","kaggle","kernels","output",kernel,
               "-p",d,"--file-pattern",r".*(?:render_report|ultra_report|h3_preflight)\.json$",
               "-o"]
    result = subprocess.run(command,env=env,text=True,capture_output=True,timeout=100)
    print("OUTPUT_DOWNLOAD_EXIT:",result.returncode)
    paths = sorted(p for p in Path(d).rglob("*.json")
                   if p.name in {"render_report.json","ultra_report.json","h3_preflight.json"})
    print("FOUND_REPORTS:", [p.name for p in paths])
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data,dict): raise ValueError("not object")
            clean = {key:sanitize(value) for key,value in data.items() if key in allow}
            print(path.name, json.dumps(clean,ensure_ascii=False))
        except (OSError,ValueError) as e:
            print(path.name,"unreadable",type(e).__name__)
