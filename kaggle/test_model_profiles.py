"""Static validation for the compact H3 project (no Kaggle GPU required)."""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for name in ("model_profiles.py", "bootstrap_h3.py", "run_from_sources.py"):
    ast.parse((ROOT / "kaggle" / name).read_text(encoding="utf-8"))

sys.path.insert(0, str(ROOT / "kaggle"))
from model_profiles import model_files, workflow_names  # noqa: E402

for mode in ("t2v", "i2v", "r2v"):
    source = json.loads((ROOT / "workflows" / "source" / (mode + ".json")).read_text(encoding="utf-8"))
    expected_type = "MiniMaxH3ReferenceToVideo" if mode == "r2v" else "MiniMaxH3ImageToVideo"
    assert any(node["class_type"] == expected_type for node in source.values())
    for profile in ("compact", "balanced", "experimental-w6a8"):
        for turbo in (True, False):
            filenames = model_files(mode, profile, turbo)
            assert all(x.endswith(".safetensors") for x in filenames.values())
            assert len(filenames) == (5 if turbo else 4)
            names = workflow_names(filenames)
            assert "ref2va" in names["diffusion_model"] if mode == "r2v" else "fl2va" in names["diffusion_model"]
            assert (names["text_encoder"].endswith("nvfp4_awq.safetensors")
                    if profile != "balanced" else
                    names["text_encoder"].endswith("int8_convrot.safetensors"))
            if profile == "experimental-w6a8":
                assert "_w6a8." in names["diffusion_model"]
            else:
                assert "_int8_convrot." in names["diffusion_model"]
            if profile == "compact":
                assert "_int8_convrot." in names["video_vae"]

nb = json.loads((ROOT / "kaggle" / "MiniMax_H3_Kaggle_T4x2_AutoVideo.ipynb").read_text(encoding="utf-8"))
assert nb["nbformat"] == 4
assert any("STORAGE_PROFILE='compact'" in "".join(c["source"]) for c in nb["cells"])
assert any("H3_STORAGE_PROFILE" in "".join(c["source"]) for c in nb["cells"])
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]))
print("PASS: all model profiles, source workflow types and Notebook syntax validated")
