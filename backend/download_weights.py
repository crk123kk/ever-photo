"""Download model weights from HuggingFace (with China mirror support).

Usage:
    python download_weights.py              # Download all available weights
    HF_ENDPOINT=https://hf-mirror.com python download_weights.py  # Use China mirror
"""
import os
import sys

import requests

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

# HuggingFace sources - China-friendly via HF_ENDPOINT mirror
HF_WEIGHTS = {
    "GFPGANv1.4.pth": ("th3w33knd/GFPGANv1.4", "GFPGANv1.4.pth"),
    # Restormer
    "real_denoising.pth": ("deepinv/Restormer", "real_denoising.pth"),
    "single_image_defocus_deblurring.pth": ("deepinv/Restormer", "single_image_defocus_deblurring.pth"),
    # DDColor
    "ddcolor_paper_tiny.pt": ("piddnad/ddcolor_paper_tiny", "pytorch_model.bin"),
}

# GitHub / direct sources - may require proxy in China
GITHUB_WEIGHTS = {
    "RealESRGAN_x2plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRGAN_x2plus.pth",
    ),
    "big-lama.pt": (
        "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
        "big-lama.pt",
    ),
    "codeformer.pth": (
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "codeformer.pth",
    ),
    "motion_deblurring.pth": (
        "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth",
        "motion_deblurring.pth",
    ),
}


def download_hf():
    for name, (repo_id, filename) in HF_WEIGHTS.items():
        target = os.path.join(WEIGHTS_DIR, name)
        if os.path.exists(target):
            size_mb = os.path.getsize(target) / 1024 / 1024
            print(f"[skip] {name} ({size_mb:.1f} MB)")
            continue
        print(f"[download] {name} from {repo_id} ...")
        try:
            hf_hub_download(repo_id, filename, local_dir=WEIGHTS_DIR)
            # If remote filename differs from local name, rename
            if name != filename:
                tmp = os.path.join(WEIGHTS_DIR, filename)
                if os.path.exists(tmp) and not os.path.exists(target):
                    os.rename(tmp, target)
            size_mb = os.path.getsize(target) / 1024 / 1024
            print(f"  -> {size_mb:.1f} MB")
        except Exception as e:
            print(f"  FAILED: {e}")


def download_github():
    for name, (url, remote_name) in GITHUB_WEIGHTS.items():
        target = os.path.join(WEIGHTS_DIR, name)
        if os.path.exists(target):
            size_mb = os.path.getsize(target) / 1024 / 1024
            print(f"[skip] {name} ({size_mb:.1f} MB)")
            continue
        print(f"[download] {name} from GitHub ...")
        print(f"  URL: {url}")
        print(f"  Note: If download fails, try using a proxy or download manually.")
        try:
            _download_with_resume(url, target)
            size_mb = os.path.getsize(target) / 1024 / 1024
            if size_mb < 10:
                print(f"  WARNING: File seems too small ({size_mb:.1f} MB), download may have failed")
                os.remove(target)
            else:
                print(f"  -> {size_mb:.1f} MB")
        except Exception as e:
            print(f"  FAILED: {e}")
            if os.path.exists(target + ".part"):
                os.remove(target + ".part")
            if os.path.exists(target):
                os.remove(target)


def _download_with_resume(url: str, target: str, chunk_size: int = 8192):
    """Download a file with resume support and progress display."""
    temp_target = target + ".part"
    headers = {}

    if os.path.exists(temp_target):
        downloaded = os.path.getsize(temp_target)
        headers["Range"] = f"bytes={downloaded}-"
        print(f"  Resuming from {downloaded / 1024 / 1024:.1f} MB ...")
    else:
        downloaded = 0

    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    if resp.status_code == 206:  # Partial content
        content_range = resp.headers.get("content-range", "")
        if "/" in content_range:
            total = int(content_range.split("/")[1])
    total += downloaded if resp.status_code == 206 else 0

    mode = "ab" if downloaded > 0 and resp.status_code == 206 else "wb"
    if resp.status_code == 200:
        mode = "wb"
        downloaded = 0

    last_print_mb = 0
    with open(temp_target, mode) as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            current_mb = int(downloaded / 1024 / 1024)
            if current_mb >= last_print_mb + 10:
                last_print_mb = current_mb
                if total > 0:
                    print(f"  {downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB")
                else:
                    print(f"  {downloaded / 1024 / 1024:.0f} MB")

    os.rename(temp_target, target)


def setup_lama_cache():
    lama_src = os.path.join(WEIGHTS_DIR, "big-lama.pt")
    lama_dst = os.path.expanduser("~/.cache/torch/hub/checkpoints/big-lama.pt")
    if os.path.exists(lama_src) and not os.path.exists(lama_dst):
        os.makedirs(os.path.dirname(lama_dst), exist_ok=True)
        import shutil
        shutil.copy2(lama_src, lama_dst)
        print(f"Copied big-lama.pt to torch hub cache")


if __name__ == "__main__":
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("=== Downloading from HuggingFace ===")
    download_hf()

    print("\n=== Downloading from GitHub ===")
    download_github()

    print("\n=== Setting up cache ===")
    setup_lama_cache()

    print("\nDone!")
    print(f"\nWeights directory: {WEIGHTS_DIR}")
    print("Available files:")
    for f in sorted(os.listdir(WEIGHTS_DIR)):
        if f.endswith((".pth", ".pt")):
            size_mb = os.path.getsize(os.path.join(WEIGHTS_DIR, f)) / 1024 / 1024
            print(f"  {f} ({size_mb:.1f} MB)")
