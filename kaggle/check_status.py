"""Read-only Kaggle worker status from GitHub Actions' 'shipin' secret."""
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_to_kaggle import load_credentials, SLUG

env, username = load_credentials()
kernel = f"{username}/{SLUG}"
print("Checking Kaggle kernel:", kernel, flush=True)
for attempt in range(7):
    proc = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "status", kernel],
        env=env, text=True, capture_output=True, timeout=45)
    msg = (proc.stdout or proc.stderr).strip()
    print(f"CHECK {attempt+1}: {msg[:1000]}", flush=True)
    if proc.returncode or any(x in msg.upper() for x in
                              ("COMPLETE", "ERROR", "CANCEL", "FAILED")):
        break
    if attempt < 6:
        time.sleep(20)
print("Kaggle UI: https://www.kaggle.com/code/" + kernel)
# List output files when complete, without exposing the credential.
if "COMPLETE" in msg.upper():
    proc = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "files", kernel],
        env=env, text=True, capture_output=True, timeout=45)
    print("Output files:", (proc.stdout or proc.stderr)[:4000], flush=True)
