"""Read-only Kaggle failure diagnosis, safe for a public GitHub Actions log.

Do not print raw notebook outputs or raw Kaggle logs: they can include secrets
and prompts even when the Kaggle kernel is private. Only emit exception types
and sanitized short exception messages.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_to_kaggle import load_credentials, SLUG

env, username = load_credentials()
kernel = f"{username}/{SLUG}"
print("Diagnosing Kaggle:", kernel, flush=True)

def cli(*args, timeout=90):
    p = subprocess.run([sys.executable, "-m", "kaggle", *args],
                       env=env, text=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout + "\n" + p.stderr

rc, raw = cli("kernels", "status", kernel)
match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", raw)
state = match.group(1) if match else "UNKNOWN"
print("KAGGLE_STATUS:", state, "CLI_EXIT:", rc, flush=True)

rc, listing = cli("kernels", "files", kernel)
# Share a count and only predictable filenames. Do not print the raw listing
# because file names could contain private information.
safe_basename = re.compile(
    r"(?i)(?:h3_preflight\.json|ultra_report\.json|render_report\.json|"
    r"ultra_render\.log|comfyui\.log|build\.log|build_config\.log|"
    r"h3_(?:t2v|i2v|r2v)_ultra_q2\.(?:mp4|webm))")
names = sorted(set(m.group(0) for m in safe_basename.finditer(listing)))
print("KNOWN_OUTPUT_FILES:", names if names else "none visible", flush=True)

if state not in {"ERROR", "FAILED", "CANCELLED", "COMPLETE", "COMPLETED"}:
    print("Notebook has not reached a terminal run state.")
    sys.exit(0)

rc, logs = cli("kernels", "logs", kernel, timeout=120)
print("KAGGLE_LOG_RETRIEVAL_EXIT:", rc, "LOG_CHARS:", len(logs), flush=True)
# Avoid emitting any line with credentials / auth info.
secret_line = re.compile(
    r"(?i)(?:token|password|secret|authorization|api[_ -]?key|bearer|"
    r"kaggle_key|kaggle_api_token|hf_token|github_token|credential)")
err_re = re.compile(
    r"(?:^|[\s:])((?:[A-Za-z_][A-Za-z_0-9]*\.)?"
    r"(?:RuntimeError|CalledProcessError|FileNotFoundError|"
    r"ModuleNotFoundError|ImportError|ValueError|KeyError|"
    r"PermissionError|OSError|AssertionError|HTTPError|"
    r"TimeoutError|ConnectionError|TypeError|MemoryError|"
    r"OutOfMemoryError|SyntaxError|NameError|Exception)"
    r"\s*:\s*[^\n]{0,220})")
failure = re.compile(
    r"(?i)\b(?:out of memory|no space left on device|disk quota exceeded|"
    r"worker died|kernel failed|cuda error|failed to allocate|"
    r"connection reset|fatal error|segmentation fault|"
    r"process exited with code|not enough memory|cannot import name)\b")
samples = []
for line in logs.splitlines():
    line = line.strip()
    if not line or secret_line.search(line):
        continue
    exc = err_re.search(line)
    frag = exc.group(1) if exc else (line if failure.search(line) else None)
    if frag:
        # Only sanitized text; exclude URL bodies and likely tokens.
        frag = re.sub(r"https?://\S+", "[URL]", frag)
        frag = re.sub(r"[A-Za-z0-9_+/\-]{45,}={0,2}", "[REDACTED]", frag)
        frag = re.sub(r"[\w.+-]+@[\w.-]+\.\w+", "[EMAIL]", frag)
        samples.append(frag[:230])
print("ERROR_SIGNATURES:")
for line in list(dict.fromkeys(samples))[-15:]:
    print("*", line, flush=True)
if not samples:
    print("* No safe Python exception signature detected in Kaggle CLI logs.")
print("Notebook URL: https://www.kaggle.com/code/" + kernel)
