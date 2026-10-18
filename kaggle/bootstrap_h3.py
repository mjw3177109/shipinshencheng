"""Kaggle MiniMax H3 mode-aware setup. This script does not read Kaggle API tokens."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import sys
import json

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

REPO = "Comfy-Org/MiniMax-H3"
MODEL = ("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
         if MODE == "r2v" else
         "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
VAE = ("vae/minimax_h3_video_vae_fp16.safetensors" if MODE == "r2v"
       else "vae/minimax_h3_video_vae_int8_convrot.safetensors")
files = [
    MODEL,
    "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    VAE,
    "vae/minimax_h3_audio_vae_fp32.safetensors",
]
if os.getenv("H3_TURBO", "1") == "1":
    files += [("loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
               if MODE == "r2v" else
               "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")]

# Use local_dir, which avoids storing a second complete model in a global HF snapshot cache.
os.environ["HF_HOME"] = str(BASE / "huggingface")
models = COMFY / "models"
needed = 0
for name in files:
    target = models / name
    if target.is_file() and target.stat().st_size > 1024 * 1024:
        continue
    metadata = get_hf_file_metadata(hf_hub_url(REPO, name), timeout=45)
    needed += metadata.size or 0
free = shutil.disk_usage(BASE).free
reserve = 6 * 1024**3
print(f"Model download estimate: {needed/1024**3:.1f} GiB; "
      f"disk free: {free/1024**3:.1f} GiB; reserve: {reserve/1024**3:.1f} GiB")
if free < needed + reserve:
    raise RuntimeError("Not enough free disk space for this mode. "
                       "Use only T2V/I2V first, clean temp storage, or attach a dataset.")

for name in files:
    print("Model:", name, flush=True)
    hf_hub_download(repo_id=REPO, filename=name, local_dir=str(models),
                    token=os.environ.get("HF_TOKEN"))
print(f"MODE={MODE} setup complete. Model files are local to {models}.")
