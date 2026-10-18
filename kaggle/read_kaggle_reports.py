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


# Only look at compiler diagnostics, never print unfiltered private build logs.
with tempfile.TemporaryDirectory() as d:
    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", kernel,
           "-p", d, "--file-pattern",
           r".*(?:build_config|build|ultra_render)\.log$", "-o"]
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=120)
    print("BUILD_LOG_DOWNLOAD_EXIT:", proc.returncode)
    files = sorted(x for x in Path(d).rglob("*.log")
                   if x.name in {"build.log","build_config.log","ultra_render.log"})
    print("BUILD_LOGS_FOUND:", [x.name for x in files])
    patterns = re.compile(
        r"(?i)(?:\berror:|\bfatal error\b|CMake Error|"
        r"Could NOT find|undefined reference|failed to|"
        r"not found|unrecognized command|unsupported gpu architecture|"
        r"nvcc fatal|No rule to make target|\[.*Error \d+\]|"
        r"missing.*dependency|ninja.*failed)")
    for f in files:
        raw = f.read_text(encoding="utf-8",errors="replace")
        keep = []
        for line in raw.splitlines():
            if patterns.search(line):
                line = sanitize(line)
                if re.search(r"(?i)(secret|token|password|authorization|key\s*[:=])",line):
                    continue
                keep.append(line[:300])
        print("LOG_DIAGNOSTICS:", f.name)
        for line in list(dict.fromkeys(keep))[-20:]:
            print("*",line)
        if not keep:
            print("* No recognized build error line. Total bytes:",len(raw))


        if f.name == "build_config.log":
            lines = raw.splitlines()
            for idx, line in enumerate(lines):
                if "CMake Error" not in line:
                    continue
                print("CMAKE_ERROR_CONTEXT:")
                for detail in lines[idx:idx + 7]:
                    if not re.search(r"(?i)(secret|password|authorization|api.?key|token\s*[:=])", detail):
                        print(">", sanitize(detail)[:300])
