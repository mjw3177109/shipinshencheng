"""Secure GitHub Actions -> Kaggle Notebook publisher.

Read token from process environment only, never print its contents. Supports
Kaggle's modern KGAT token and legacy Kaggle JSON credentials.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import requests

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle" / "MiniMax_H3_Kaggle_T4x2_AutoVideo.ipynb"
SLUG = "minimax-h3-t4x2-shipin"
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def load_credentials() -> tuple[dict[str, str], str]:
    token = os.getenv("SHIPIN_KAGGLE_CREDENTIAL", "").strip()
    if not token:
        raise RuntimeError(
            "Missing GitHub Actions secret 'shipin'. "
            "Add Kaggle API Token to this repository's Actions secrets, "
            "or check whether it was instead stored in Kaggle Notebook secrets."
        )
    env = dict(os.environ)
    username = os.getenv("KAGGLE_USERNAME_OVERRIDE", "").strip()

    if token.startswith("{"):
        try:
            credentials = json.loads(token)
        except json.JSONDecodeError as err:
            raise RuntimeError("Secret 'shipin' is invalid legacy Kaggle JSON.") from err
        legacy_user = credentials.get("username", "")
        legacy_key = credentials.get("key", "")
        if not legacy_user or not legacy_key:
            raise RuntimeError("Legacy Kaggle JSON must contain username and key.")
        username = username or legacy_user
        env.pop("KAGGLE_API_TOKEN", None)
        env["KAGGLE_USERNAME"] = legacy_user
        env["KAGGLE_KEY"] = legacy_key
    else:
        env["KAGGLE_API_TOKEN"] = token
        # The Kaggle documented OAuth introspection endpoint reports username;
        # some personal API token variants may not introspect as OAuth tokens.
        if not username:
            try:
                response = requests.post(
                    "https://www.kaggle.com/api/v1/oauth2/introspect",
                    json={"token": token}, timeout=20,
                )
                response.raise_for_status()
                identity = response.json()
                if identity.get("active") is True:
                    username = identity.get("username", "")
            except (requests.RequestException, ValueError):
                pass
        if not username:
            raise RuntimeError(
                "Unable to resolve Kaggle username from token. "
                "Supply username via 'kaggle_username' workflow input; "
                "no token is displayed or persisted."
            )
    if not USERNAME_PATTERN.fullmatch(username):
        raise RuntimeError("Kaggle username format is invalid.")
    return env, username


def publish() -> None:
    if not NOTEBOOK.is_file():
        raise RuntimeError("Source notebook is missing.")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4 or not nb.get("cells"):
        raise RuntimeError("Notebook has invalid or empty structure.")
    env, username = load_credentials()
    print(f"Kaggle target: {username}/{SLUG} (private)")
    mode = os.getenv("KAGGLE_RUN_VIDEO", "true").lower()
    if mode not in ("true", "false"):
        raise RuntimeError("KAGGLE_RUN_VIDEO must be true or false")

    with tempfile.TemporaryDirectory(prefix="kaggle_h3_publish_") as folder:
        dest = Path(folder)
        shutil.copy2(NOTEBOOK, dest / NOTEBOOK.name)
        metadata = {
            "id": f"{username}/{SLUG}",
            "title": "MiniMax H3 T4x2 Auto Video",
            "code_file": NOTEBOOK.name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (dest / "kernel-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        command = [
            sys.executable, "-m", "kaggle", "kernels", "push", "-p",
            str(dest), "--accelerator", "NvidiaTeslaT4"
        ]
        if mode == "false":
            command.append("--no-run")
        print("Publishing Kaggle notebook using the Kaggle CLI...")
        subprocess.run(command, env=env, check=True, timeout=600)
        print(f"Kaggle URL: https://www.kaggle.com/code/{username}/{SLUG}")
        print("Submitted notebook. Kaggle GPU inference status is not verified.")


if __name__ == "__main__":
    publish()
