"""Static tests for Ultra-Q2 code (no GPU, no downloads, no credentials)."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "kaggle" / "ultra_gguf_h3.py"
text = MAIN.read_text(encoding="utf-8")
ast.parse(text, filename=str(MAIN))
assert "stable-diffusion.cpp" in text
assert "diffusion=cuda0&cuda1" in text, "No real dual GPU backend assignment"
assert "H3_PREFLIGHT_ONLY" in text
assert "H3_SINGLE_GPU" in text
assert "qwen3vl-32B-MiniMax-H3-Q2_K.gguf" in text
assert "minimax_h3_video_vae_fp16.safetensors" in text
assert "minimax_h3_audio_vae_fp32.safetensors" in text
assert "minimax_h3_ref2va_pruned-Q2_K.gguf" in text
assert "minimax_h3_fl2va_pruned-Q2_K.gguf" in text
assert "get_hf_file_metadata" in text
assert "h3_inputs" in text
for mode, typ in (("t2v", "MiniMaxH3ImageToVideo"),
                  ("i2v", "MiniMaxH3ImageToVideo"),
                  ("r2v", "MiniMaxH3ReferenceToVideo")):
    data = json.loads((ROOT / "workflows" / "source" / f"{mode}.json").read_text(encoding="utf-8"))
    assert sum(x.get("class_type") == typ for x in data.values()) == 1
    target = next(x["inputs"] for x in data.values() if x.get("class_type") == typ)
    if mode == "i2v":
        assert isinstance(target.get("first_frame"), list)
    if mode == "r2v":
        for n in range(3):
            assert isinstance(target.get(f"ref_images.ref_image_{n}"), list)
nb = json.loads((ROOT / "kaggle" / "MiniMax_H3_Kaggle_T4x2_AutoVideo.ipynb").read_text(encoding="utf-8"))
assert nb["nbformat"] == 4
assert any("BACKEND='ultra-q2'" in "".join(c["source"]) for c in nb["cells"])
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print("PASS Ultra Q2: source workflows, dual GPU flag, no-token preflight, Notebook Python syntax")
