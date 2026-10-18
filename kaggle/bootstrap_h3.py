# Kaggle H3 video bootstrap
import os,subprocess,sys,json,time,uuid,shutil
from pathlib import Path
def sh(*a): subprocess.run(list(a),check=True)
assert shutil.which("nvidia-smi"),"Kaggle GPU required"
sh("nvidia-smi")
import torch
assert torch.cuda.device_count()>=2,"Select Kaggle GPU T4 x2"
base=Path("/kaggle/temp/minimax_h3");base.mkdir(parents=True,exist_ok=True)
comfy=base/"ComfyUI"
if not (comfy/"main.py").exists(): sh("git","clone","--depth=1","https://github.com/Comfy-Org/ComfyUI.git",str(comfy))
sh(sys.executable,"-m","pip","install","-q","-r",str(comfy/"requirements.txt"))
sh(sys.executable,"-m","pip","install","-q","huggingface_hub")
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import hf_hub_download
files=[
("Comfy-Org/MiniMax-H3","diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",comfy/"models"),
("Comfy-Org/MiniMax-H3","text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",comfy/"models"),
("Comfy-Org/MiniMax-H3","vae/minimax_h3_video_vae_int8_convrot.safetensors",comfy/"models"),
("Comfy-Org/MiniMax-H3","vae/minimax_h3_audio_vae_fp32.safetensors",comfy/"models"),
("Comfy-Org/MiniMax-H3","loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",comfy/"models"),
]
print("These are large files. Check Kaggle disk capacity before running.")
if shutil.disk_usage(base).free<48_000_000_000: raise RuntimeError("Need around 48 GB free on download filesystem")
for repo,filename,folder in files:
    print("Download",repo,filename,flush=True)
    hf_hub_download(repo_id=repo,filename=filename,local_dir=str(folder),token=os.environ.get("HF_TOKEN"))
print("Models downloaded. Run the next cells to start ComfyUI.")
