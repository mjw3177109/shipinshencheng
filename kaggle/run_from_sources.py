"""Compile the three user-supplied H3 API graphs for small Kaggle test renders.

Keep modes distinct: T2V/I2V use FL2VA, R2V uses Ref2VA. Standard ComfyUI
currently runs each render on GPU0, not sharded over the two T4 devices.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import requests
from model_profiles import model_files, selected_profile, workflow_names

ROOT = Path(__file__).resolve().parents[1]
MODE = os.environ.get("H3_MODE", "t2v").lower()
if MODE not in ("t2v", "i2v", "r2v"):
    raise ValueError("H3_MODE must be t2v, i2v, or r2v")
SOURCE = ROOT / "workflows" / "source" / (MODE + ".json")
COMFY = Path("/kaggle/temp/minimax_h3/ComfyUI")
OUT = Path("/kaggle/working/h3_videos")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "comfyui.log"
REPORT = OUT / "render_report.json"
API = "http://127.0.0.1:8188"
TURBO = os.environ.get("H3_TURBO", "1") == "1"
WIDTH = int(os.environ.get("H3_WIDTH", 384))
HEIGHT = int(os.environ.get("H3_HEIGHT", 384 if MODE == "i2v" else 224))
FRAMES = int(os.environ.get("H3_FRAMES", 22))
if WIDTH % 32 or HEIGHT % 32 or min(WIDTH, HEIGHT) < 128:
    raise ValueError("Width/height must be multiples of 32 and >= 128")
if FRAMES < 22 or (FRAMES - 5) % 17:
    raise ValueError("MiniMax H3 length must follow 17*k+5; try 22 or 124 frames")
source = json.loads(SOURCE.read_text(encoding="utf-8"))
original = next(v for v in source.values()
                if v["class_type"] in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"))
if MODE == "r2v" and original["class_type"] != "MiniMaxH3ReferenceToVideo":
    raise ValueError("R2V source workflow does not use reference-video node")
if MODE != "r2v" and original["class_type"] != "MiniMaxH3ImageToVideo":
    raise ValueError("T2V/I2V source workflow uses the wrong model type")

def original_prompt():
    inp = original["inputs"]["prompt"]
    if isinstance(inp, str):
        return inp
    return source[inp[0]]["inputs"]["value"]

def get_image_names():
    if MODE == "t2v":
        return []
    if MODE == "i2v":
        return [next(v["inputs"]["image"] for v in source.values()
                     if v["class_type"] == "LoadImage")]
    ref_items = []
    for i in range(3):
        v = original["inputs"].get(f"ref_images.ref_image_{i}")
        if not isinstance(v, list):
            raise ValueError(f"Missing reference image pin {i}")
        ref_items.append(source[v[0]]["inputs"]["image"])
    return ref_items

# Never silently use fabricated reference pictures. Supply the exact filenames
# under /kaggle/working/h3_inputs before running image/reference workflows.
images = get_image_names()
input_root = Path("/kaggle/working/h3_inputs")
for name in images:
    p = input_root / name
    if not p.is_file():
        raise FileNotFoundError(f"{MODE.upper()} needs reference input: {p}")
    (COMFY / "input").mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, COMFY / "input" / name)

PROFILE = selected_profile()
name_map = workflow_names(model_files(MODE, PROFILE, TURBO))
model = name_map["diffusion_model"]
video_vae = name_map["video_vae"]
text_encoder = name_map["text_encoder"]
lora = name_map.get("turbo_lora")
steps = (4 if MODE == "r2v" else 8) if TURBO else 20

P = {}
def node(id_, typ, **inputs):
    P[str(id_)] = {"class_type": typ, "inputs": inputs}

node(1, "UNETLoader", unet_name=model, weight_dtype="default")
node(2, "CLIPLoader", clip_name=text_encoder,
     type="minimax", device="default")
node(3, "VAELoader", vae_name=video_vae)
node(4, "VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
if TURBO:
    node(5, "LoraLoaderModelOnly", model=["1", 0],
         lora_name=lora, strength_model=1.0)
model_ref = ["5" if TURBO else "1", 0]
src_input = {"prompt": os.environ.get("H3_PROMPT") or original_prompt(),
             "width": WIDTH, "height": HEIGHT, "length": FRAMES,
             "clip": ["2", 0], "vae": ["3", 0]}
if MODE == "i2v":
    node(17, "LoadImage", image=images[0])
    src_input["first_frame"] = ["17", 0]
if MODE == "r2v":
    src_input["audio_vae"] = ["4", 0]
    src_input["ref_image_size"] = original["inputs"].get("ref_image_size", "match")
    for idx, name in enumerate(images):
        node(17 + idx, "LoadImage", image=name)
        src_input[f"ref_images.ref_image_{idx}"] = [str(17 + idx), 0]
    node(6, "MiniMaxH3ReferenceToVideo", **src_input)
else:
    node(6, "MiniMaxH3ImageToVideo", **src_input)
node(7, "RandomNoise", noise_seed=int(os.environ.get("H3_SEED", 123456)))
node(8, "BasicGuider", model=model_ref, conditioning=["6", 0])
sampler_name = ("euler" if TURBO else "res_multistep")
node(9, "KSamplerSelect", sampler_name=sampler_name)
node(10, "BasicScheduler", model=model_ref, scheduler="simple",
     steps=steps, denoise=1.0)
node(11, "SamplerCustomAdvanced", noise=["7", 0], guider=["8", 0],
     sampler=["9", 0], sigmas=["10", 0], latent_image=["6", 1])
node(12, "VAEDecode", samples=["11", 0], vae=["3", 0])
node(13, "VAEDecodeAudio", samples=["11", 0], vae=["4", 0])
node(14, "CreateVideo", images=["12", 0], audio=["13", 0],
     fps=24.0, bit_depth=8)
node(15, "SaveVideo", video=["14", 0],
     filename_prefix=f"video/H3_{MODE.upper()}",
     format="auto", codec="auto")

status = {
    "mode": MODE, "turbo": TURBO, "steps": steps,
    "width": WIDTH, "height": HEIGHT, "frames": FRAMES,
    "gpu_strategy": "GPU0 single-device inference; GPU1 not combined",
    "source_workflow": str(SOURCE.relative_to(ROOT)),
    "storage_profile": PROFILE,
    "selected_model_files": name_map,
    "reference_images": images, "status": "pending",
}
def persist():
    REPORT.write_text(json.dumps(status, ensure_ascii=False, indent=2),
                      encoding="utf-8")

persist()
env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = "0"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
with LOG.open("w", encoding="utf-8") as fp:
    server = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188",
         "--lowvram", "--disable-auto-launch", "--output-directory", str(OUT)],
        cwd=COMFY, env=env, stdout=fp, stderr=subprocess.STDOUT)
try:
    for attempt in range(180):
        if server.poll() is not None:
            raise RuntimeError("ComfyUI exited early; inspect comfyui.log")
        try:
            info_resp = requests.get(API + "/object_info", timeout=3)
            if info_resp.ok:
                info = info_resp.json()
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        raise TimeoutError("ComfyUI API startup timeout (6 minutes)")
    required = {"UNETLoader", "CLIPLoader", "VAELoader",
                "SamplerCustomAdvanced", "CreateVideo", "SaveVideo",
                "MiniMaxH3ReferenceToVideo" if MODE == "r2v"
                else "MiniMaxH3ImageToVideo"}
    missing = sorted(required.difference(info))
    if missing:
        raise RuntimeError("Your ComfyUI version lacks nodes: " + ", ".join(missing))
    req = requests.post(
        API + "/prompt", json={"prompt": P, "client_id": str(uuid.uuid4())},
        timeout=120)
    if not req.ok:
        raise RuntimeError("ComfyUI graph validation: " + req.text[:5000])
    prompt_id = req.json()["prompt_id"]
    status["prompt_id"] = prompt_id
    status["status"] = "running"
    persist()
    print("Submitted:", prompt_id, flush=True)
    for n in range(1440):
        history = requests.get(API + "/history/" + prompt_id, timeout=30).json()
        if prompt_id in history:
            result = history[prompt_id]
            if result.get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(result["status"], ensure_ascii=False)[:4000])
            if result.get("status", {}).get("completed") is False:
                raise RuntimeError("ComfyUI run did not complete; inspect history/log")
            break
        if server.poll() is not None:
            raise RuntimeError("ComfyUI process ended unexpectedly")
        if n % 12 == 0:
            print(f"Running {n*5}s", flush=True)
        time.sleep(5)
    else:
        raise TimeoutError("Video render exceeded two hours")
    videos = sorted(OUT.rglob("*.mp4"), key=lambda x: x.stat().st_mtime)
    if not videos:
        raise RuntimeError("History completed but no MP4 exists")
    status["status"] = "succeeded"
    status["outputs"] = [str(v) for v in videos]
    persist()
    print("Videos:", status["outputs"], flush=True)
except BaseException as error:
    status["status"] = "failed"
    status["error"] = str(error)
    persist()
    print(f"FAILED: {error}", file=sys.stderr, flush=True)
    raise
finally:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
