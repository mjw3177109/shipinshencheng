"""Ultra-small MiniMax H3 GGUF for Kaggle T4 x2.

This is an ALTERNATIVE inference backend, not a ComfyUI loader replacement.
It consumes the 3 user-provided API JSONs only for prompt and image references.
Actual inference is stable-diffusion.cpp. No model files enter GitHub.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/kaggle/working")
BASE = Path("/kaggle/temp/h3_gguf_ultra")
MODEL_DIR = BASE / "models"
SRC = BASE / "stable-diffusion.cpp"
BIN = SRC / "build" / "bin" / "sd-cli"
OUT = WORK / "h3_videos"
REPORT = OUT / "ultra_report.json"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODE = os.getenv("H3_MODE", "t2v").lower()
if MODE not in ("t2v", "i2v", "r2v"):
    raise ValueError(f"Invalid H3 mode: {MODE}")
W = int(os.getenv("H3_WIDTH", "384"))
H = int(os.getenv("H3_HEIGHT", "224"))
FRAMES = int(os.getenv("H3_FRAMES", "22"))
STEPS = int(os.getenv("H3_ULTRA_STEPS", "20"))
if W < 128 or H < 128 or W % 32 or H % 32:
    raise ValueError("H3 width and height must be >=128 and divisible by 32")
if FRAMES < 22 or (FRAMES - 5) % 17:
    raise ValueError("H3 frames must equal 17*k+5, e.g. 22, 39, 56, 124")
if not 1 <= STEPS <= 40:
    raise ValueError("H3_ULTRA_STEPS must be between 1 and 40")
DRY_RUN = os.getenv("H3_PREFLIGHT_ONLY", "0") == "1"
SINGLE_CARD = os.getenv("H3_SINGLE_GPU", "0") == "1"

diffusion_name = (
    "minimax_h3_ref2va_pruned-Q2_K.gguf" if MODE == "r2v"
    else "minimax_h3_fl2va_pruned-Q2_K.gguf"
)
source_specs = [
    ("unsloth/MiniMax-H3-GGUF", diffusion_name),
    ("realrebelai/MiniMax-H3_GGUFs", "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"),
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors"),
    ("Comfy-Org/MiniMax-H3", "vae/minimax_h3_audio_vae_fp32.safetensors"),
]

info = {
    "mode": MODE, "backend": "stable-diffusion.cpp-GGUF",
    "quant": "Q2_K",
    "diffusion_checkpoint": diffusion_name,
    "t2v_i2v_r2v_mode_locked": True,
    "width": W, "height": H, "frames": FRAMES, "steps": STEPS,
    "phase": "initializing", "success": False,
    "notes": ["GGUF Q2 quality is lower than INT8; visual fidelity is not guaranteed.",
              "Dual GPU layer splitting is supported by backend but not validated on this Kaggle T4x2 H3 run."],
}
def save():
    REPORT.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
save()

def invoke(args, cwd=None, timeout=None, logfile=None):
    if logfile is not None:
        with Path(logfile).open("w", encoding="utf-8", errors="replace") as f:
            subprocess.run(args, cwd=cwd, check=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    else:
        subprocess.run(args, cwd=cwd, check=True, timeout=timeout)

def get_source():
    path = ROOT / "workflows" / "source" / (MODE + ".json")
    graph = json.loads(path.read_text(encoding="utf-8"))
    cls = "MiniMaxH3ReferenceToVideo" if MODE == "r2v" else "MiniMaxH3ImageToVideo"
    candidates = [(k, v) for k, v in graph.items() if v.get("class_type") == cls]
    if len(candidates) != 1:
        raise ValueError(f"{MODE} workflow must have exactly one {cls}, found {len(candidates)}")
    inp = candidates[0][1]["inputs"]
    text = inp["prompt"]
    if isinstance(text, list):
        text = graph[text[0]]["inputs"]["value"]
    image_args = []
    inputdir = WORK / "h3_inputs"
    if MODE == "i2v":
        first = inp.get("first_frame")
        if not isinstance(first, list):
            raise ValueError("Original I2V JSON has no first_frame connection")
        name = graph[first[0]]["inputs"]["image"]
        path = inputdir / name
        if not path.is_file():
            raise FileNotFoundError(f"I2V requires uploaded initial image: {path}")
        image_args.extend(["--init-img", str(path)])
    elif MODE == "r2v":
        for i in range(3):
            ref = inp.get(f"ref_images.ref_image_{i}")
            if not isinstance(ref, list):
                raise ValueError(f"Reference {i} not connected in original R2V")
            name = graph[ref[0]]["inputs"]["image"]
            path = inputdir / name
            if not path.is_file():
                raise FileNotFoundError(f"R2V needs uploaded image: {path}")
            image_args.extend(["--ref-image", str(path)])
    return os.getenv("H3_PROMPT") or text, image_args

def free_gib(path):
    return shutil.disk_usage(path).free / 1024**3

try:
    import requests
    from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata
except ImportError as e:
    if DRY_RUN:
        raise RuntimeError("Preflight requires requests and huggingface_hub") from e
    invoke([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.28", "requests", "psutil"])
    from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata

prompt, image_args = get_source()  # Fail before any large download if required images missing.
info["input_images"] = image_args[1::2]
if not shutil.which("nvidia-smi"):
    raise RuntimeError("Enable GPU T4 x2 in Kaggle notebook Settings")
gpu_txt = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
gpus = sum(1 for ln in gpu_txt.splitlines() if ln.startswith("GPU "))
if not gpus:
    raise RuntimeError("No NVIDIA GPU detected")
dual = gpus >= 2 and not SINGLE_CARD
info["gpu_count"] = gpus
info["gpu_strategy"] = "diffusion=cuda0&cuda1 layer-split" if dual else "diffusion=cuda0"
info["gpu_inventory"] = gpu_txt.strip()
print(gpu_txt, flush=True)

# Calculate EXACT remote file sizes before downloading; no shadow HF cache snapshot.
items = []
missing_bytes = 0
for repo, filename in source_specs:
    dest = MODEL_DIR / Path(filename).name
    meta = get_hf_file_metadata(hf_hub_url(repo, filename), timeout=60)
    if not meta.size:
        raise RuntimeError(f"Cannot resolve model size: {repo}/{filename}")
    valid = dest.is_file() and dest.stat().st_size == meta.size
    items.append({"repo": repo, "filename": filename, "bytes": meta.size,
                  "present": valid, "dest": str(dest)})
    if not valid:
        missing_bytes += meta.size
info["models"] = [{k: v for k, v in d.items() if k != "dest"} for d in items]
info["model_total_gib"] = round(sum(d["bytes"] for d in items)/1024**3, 2)
info["download_missing_gib"] = round(missing_bytes/1024**3, 2)
info["free_scratch_gib"] = round(free_gib(BASE), 2)
info["phase"] = "preflight"
save()
print(f"Ultra-Q2 total {info['model_total_gib']:.2f} GiB, missing "
      f"{info['download_missing_gib']:.2f} GiB, scratch free "
      f"{info['free_scratch_gib']:.2f} GiB", flush=True)

# Scratch needs room for the GGUF models plus source compilation and temp files.
build_reserve = 7 * 1024**3
if shutil.disk_usage(BASE).free < missing_bytes + build_reserve:
    raise RuntimeError(f"Not enough scratch space: require at least "
                       f"{(missing_bytes + build_reserve)/1024**3:.1f} GiB free. "
                       "Do not put 20+ GB GGUF weights in the persistent /kaggle/working directory.")
if DRY_RUN:
    print("DRY RUN PASS: models and disk checked. No heavy files were downloaded.")
    info["phase"] = "preflight_passed"
    save()
    sys.exit(0)

# For Kaggle, these downloads are ephemeral and do not count toward saved Output.
os.environ["HF_HOME"] = str(BASE / "hf_cache")
info["phase"] = "downloading"
save()
for obj in items:
    if not obj["present"]:
        print("Downloading:", obj["filename"], flush=True)
        hf_hub_download(repo_id=obj["repo"], filename=obj["filename"],
                        local_dir=str(MODEL_DIR), token=os.getenv("HF_TOKEN"))
        p = MODEL_DIR / obj["filename"]
        dest = Path(obj["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if p != dest:
            shutil.move(str(p), str(dest))
        if dest.stat().st_size != obj["bytes"]:
            raise RuntimeError(f"Incomplete download: {dest}")

# Build a CUDA 7.5 native backend, rather than ComfyUI GGUF-H3, which
# explicitly does not support _K diffusion quants efficiently.
info["phase"] = "building"
save()
if not BIN.is_file():
    if not shutil.which("cmake") or not shutil.which("git"):
        raise RuntimeError("Kaggle image lacks cmake or git")
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).is_file():
        raise RuntimeError("CUDA Toolkit nvcc missing. A CUDA build (not CPU) is required.")
    if not (SRC / "CMakeLists.txt").exists():
        invoke(["git", "clone", "--depth", "1", "--recurse-submodules",
                "--shallow-submodules", "https://github.com/leejet/stable-diffusion.cpp.git",
                str(SRC)], timeout=600)
    build = SRC / "build"
    invoke(["cmake", "-S", str(SRC), "-B", str(build),
            "-DSD_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CUDA_ARCHITECTURES=75",
            "-DGGML_NATIVE=OFF",
            f"-DCMAKE_CUDA_COMPILER={nvcc}"],
           timeout=600, logfile=OUT / "build_config.log")
    invoke(["cmake", "--build", str(build), "-j", str(min(os.cpu_count() or 2, 4))],
           timeout=3000, logfile=OUT / "build.log")
    if not BIN.is_file():
        raise RuntimeError("Built successfully but sd-cli executable missing")

# Verify flags before expensive inference; fail on incompatible upstream changes.
cli_help = subprocess.run([str(BIN), "--help"], capture_output=True, text=True, timeout=30)
opts = (cli_help.stdout + cli_help.stderr)
required_flags = ["--diffusion-model", "--llm", "--video-frames",
                  "--backend", "--ref-image", "--cfg-scale"]
missing_flags = [x for x in required_flags if x not in opts]
if missing_flags:
    raise RuntimeError(f"sd-cli build incompatible with H3 script: {missing_flags}")

files = {Path(it["filename"]).name: Path(it["dest"]) for it in items}
video = OUT / f"h3_{MODE}_ultra_q2.webm"
command = [
    str(BIN), "--mode", "vid_gen",
    "--diffusion-model", str(files[diffusion_name]),
    "--llm", str(files["qwen3vl-32B-MiniMax-H3-Q2_K.gguf"]),
    "--vae", str(files["minimax_h3_video_vae_fp16.safetensors"]),
    "--audio-vae", str(files["minimax_h3_audio_vae_fp32.safetensors"]),
    "--prompt", prompt,
    "--width", str(W), "--height", str(H),
    "--video-frames", str(FRAMES), "--fps", "24",
    "--steps", str(STEPS), "--cfg-scale", "1.0",
    "--rng", "cpu", "--flash-attn", "auto",
    "--backend", (
        "diffusion=cuda0&cuda1,te=cpu,vae=cuda0"
        if dual else "diffusion=cuda0,te=cpu,vae=cuda0"
    ),
    "--auto-fit", "off", "--output", str(video),
] + image_args
# No secret/token in command line or logs.
info["phase"] = "rendering"
info["backend_assignment"] = command[command.index("--backend") + 1]
info["render_log"] = str(OUT / "ultra_render.log")
save()
print("Generating:", MODE, W, "x", H, FRAMES, "frames", info["backend_assignment"], flush=True)
start = time.monotonic()
try:
    invoke(command, timeout=3*3600, logfile=OUT / "ultra_render.log")
    if not video.exists() or video.stat().st_size < 1024:
        raise RuntimeError(f"sd-cli exited without a nonempty video: {video}")
    info["phase"] = "video_created"
    info["webm_path"] = str(video)
    info["webm_bytes"] = video.stat().st_size
    if shutil.which("ffmpeg"):
        mp4 = video.with_suffix(".mp4")
        invoke(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video), "-c:v", "libx264", "-crf", "20",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                str(mp4)], timeout=600, logfile=OUT / "ffmpeg.log")
        if mp4.exists() and mp4.stat().st_size > 1024:
            info["mp4_path"] = str(mp4)
    info["success"] = True
except Exception as exc:
    info["error"] = f"{type(exc).__name__}: {exc}"
    raise
finally:
    info["elapsed_seconds"] = int(time.monotonic() - start)
    save()
print("Ultra-Q2 output:", info.get("mp4_path", str(video)), flush=True)
