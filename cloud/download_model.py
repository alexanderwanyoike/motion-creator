"""
Model download script for HY-Motion-1.0
Run this to pre-download model weights to the network volume
"""

import os
from huggingface_hub import snapshot_download

MODEL_REPO = "Tencent-Hunyuan/HY-Motion-1.0"
MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/hy-motion-1.0")


def download_model():
    """Download HY-Motion-1.0 model from HuggingFace."""
    print(f"Downloading HY-Motion-1.0 to {MODEL_PATH}...")

    os.makedirs(MODEL_PATH, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_PATH,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Model downloaded successfully to {MODEL_PATH}")


if __name__ == "__main__":
    download_model()
