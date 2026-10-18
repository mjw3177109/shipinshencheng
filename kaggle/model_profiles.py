"""MiniMax H3 model file profiles shared by setup and ComfyUI API runner.

Keep storage precision separate from workflow type. Neither profile changes
T2V/I2V <-> R2V; only file selection differs.
"""
from __future__ import annotations

import os

PROFILE_NAMES = ("compact", "balanced", "experimental-w6a8")
MODES = ("t2v", "i2v", "r2v")
REPO_ID = "Comfy-Org/MiniMax-H3"

def selected_profile() -> str:
    name = os.environ.get("H3_STORAGE_PROFILE", "compact").strip().lower()
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown H3_STORAGE_PROFILE={name!r}; choose {PROFILE_NAMES}")
    return name

def model_files(mode: str, profile: str | None = None, turbo: bool = True) -> dict[str, str]:
    profile = profile or selected_profile()
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown H3 storage profile: {profile}")
    if mode not in MODES:
        raise ValueError(f"Unknown H3 mode: {mode}")
    prefix = "ref2va" if mode == "r2v" else "fl2va"
    suffix = "pruned_w6a8" if profile == "experimental-w6a8" else "pruned_int8_convrot"
    text_encoder = ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
                    if profile == "balanced" else
                    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    video_vae = ("minimax_h3_video_vae_fp16.safetensors"
                 if mode == "r2v" and profile == "balanced" else
                 "minimax_h3_video_vae_int8_convrot.safetensors")
    lora = ("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
            if mode == "r2v" else
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")
    files = {
        "diffusion_model": f"diffusion_models/minimax_h3_{prefix}_{suffix}.safetensors",
        "text_encoder": f"text_encoders/{text_encoder}",
        "video_vae": f"vae/{video_vae}",
        "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
    }
    if turbo:
        files["turbo_lora"] = f"loras/{lora}"
    return files

def workflow_names(files: dict[str, str]) -> dict[str, str]:
    return {key: file.rsplit("/", 1)[-1] for key, file in files.items()}
