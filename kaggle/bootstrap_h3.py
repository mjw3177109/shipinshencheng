"""Kaggle MiniMax H3 mode-aware setup. This script does not read Kaggle API tokens."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import sys
import json
from model_profiles import REPO_ID, model_files, selected_profile

MODES = ("t2v", "i2v", "r2v")
MODE = os.environ.get("H3_MODE", "t2v").lower()
if MODE not in MODES:
    raise SystemExit(f"Invalid H3_MODE: {MODE}")
if not shutil.which("nvidia-smi"):
    raise RuntimeError("Kaggle GPU is not enabled")
subprocess.run(["nvidia-smi", "-L"], check=True)
try:
    import torch
    devices = torch.cuda.device_count()
    print(f"Detected {devices} CUDA GPUs. Standard ComfyUI video generation uses GPU 0 only.")
    if devices < 2:
        print("WARNING: Kaggle T4 x2 was requested, but fewer than 2 CUDA GPUs detected.")
except ImportError:
    print("PyTorch GPU count not yet available")

BASE = Path("/kaggle/temp/minimax_h3")
BASE.mkdir(parents=True, exist_ok=True)
COMFY = BASE / "ComfyUI"
if not (COMFY / "main.py").exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Comfy-Org/ComfyUI.git", str(COMFY)],
                   check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(COMFY / "requirements.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "huggingface_hub>=0.28", "requests", "psutil", "pillow"], check=True)

from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata

# Persist non-sensitive hardware facts before downloading large checkpoints.
import psutil
probe = {
    "mode": MODE,
    "cpu_ram_gib": round(psutil.virtual_memory().total / 1024**3, 2),
    "available_ram_gib": round(psutil.virtual_memory().available / 1024**3, 2),
    "scratch_free_gib": round(shutil.disk_usage(BASE).free / 1024**3, 2),
    "gpu_count": int(devices) if "devices" in locals() else None,
    "notes": [
        "ComfyUI inference currently uses GPU0 only.",
        "Kaggle T4x2 host RAM can be below the 32-64GiB low-VRAM reference.",
        "An overlay filesystem may over-report true remaining writable disk.",
    ],
}
Path("/kaggle/working/h3_preflight.json").write_text(
    json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
print("Kaggle resource preflight:", json.dumps(probe, ensure_ascii=False), flush=True)


PROFILE = selected_profile()
TURBO = os.environ.get("H3_TURBO", "1") == "1"
filemap = model_files(MODE, PROFILE, TURBO)
files = list(filemap.values())
print(f"H3 storage profile={PROFILE} mode={MODE} turbo={TURBO}", flush=True)
if PROFILE == "experimental-w6a8":
    print("WARNING: W6A8 is experimental on T4 and can require newer CUDA/ComfyUI.", flush=True)

# Do not fetch all model variants. T2V and I2V share the same FL2VA files.
# R2V downloads a separate diffusion model only when selected.
os.environ["HF_HOME"] = str(BASE / "huggingface")
models = COMFY / "models"
needed = 0
details = []
for name in files:
    target = models / name
    metadata = get_hf_file_metadata(hf_hub_url(REPO_ID, name), timeout=45)
    remote_bytes = int(metadata.size or 0)
    if remote_bytes <= 0:
        raise RuntimeError(f"Hugging Face did not provide file size: {name}")
    exists = target.is_file() and target.stat().st_size == remote_bytes
    remaining = 0 if exists else remote_bytes
    needed += remaining
    details.append({"name": name, "remote_gb": round(remote_bytes / 1e9, 2),
                    "already_present": exists})
free = shutil.disk_usage(BASE).free
reserve = 6 * 1024**3
probe.update({
    "storage_profile": PROFILE,
    "expected_download_gib": round(needed / 1024**3, 2),
    "required_free_with_reserve_gib": round((needed + reserve) / 1024**3, 2),
    "downloads": details,
})
Path("/kaggle/working/h3_preflight.json").write_text(
    json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Model download estimate: {needed/1024**3:.1f} GiB; "
      f"disk free: {free/1024**3:.1f} GiB; reserve: {reserve/1024**3:.1f} GiB")
if free < needed + reserve:
    raise RuntimeError(
        f"Insufficient scratch storage for {MODE}/{PROFILE}: need "
        f"{(needed + reserve)/1024**3:.1f} GiB free, "
        f"found {free/1024**3:.1f} GiB. "
        "Switch to a smaller profile or attach a supported model dataset. "
        "Never assume 20GB of saved /kaggle/working storage can hold H3."
    )

for name in files:
    print("Model:", name, flush=True)
    hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=str(models),
                    token=os.environ.get("HF_TOKEN"))
print(f"MODE={MODE} PROFILE={PROFILE} ready. Model files: {models}.")
