"""
RunPod Serverless Handler for HY-Motion-1.0

Generates SMPL-H motion data from text prompts using the official HY-Motion inference code.
"""

import os
import sys
import io
import base64
import time
from typing import Any

import numpy as np
import runpod

# Add HY-Motion to path
sys.path.insert(0, "/app/hy-motion")

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/HY-Motion-1.0")
MODEL_REPO = "tencent/HY-Motion-1.0"
DEFAULT_FPS = 30

# Global runtime reference
runtime = None


def download_model_if_needed():
    """Download model from HuggingFace if not present."""
    from huggingface_hub import snapshot_download

    # Check if model exists by looking for config.yml
    config_path = os.path.join(MODEL_PATH, "config.yml")
    if os.path.exists(config_path):
        print(f"Model already exists at {MODEL_PATH}")
        return MODEL_PATH

    print(f"Downloading {MODEL_REPO} to {MODEL_PATH}...")
    os.makedirs(MODEL_PATH, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_PATH,
        local_dir_use_symlinks=False,
    )
    print("Download complete!")
    return MODEL_PATH


def load_model():
    """Load HY-Motion-1.0 runtime."""
    global runtime

    if runtime is not None:
        return runtime

    model_path = download_model_if_needed()

    print(f"Loading HY-Motion-1.0 from {model_path}...")
    start_time = time.time()

    # Import HY-Motion runtime
    from hymotion.utils.t2m_runtime import T2MRuntime

    # Initialize runtime
    config_path = os.path.join(model_path, "config.yml")
    ckpt_name = "latest.ckpt"

    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_name,
        device_ids=[0],  # Use first GPU
        prompt_engineering_model_path=None,
        prompt_engineering_host=None,
    )

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")

    return runtime


def generate_motion(
    prompt: str,
    duration: float = 4.0,
    fps: int = DEFAULT_FPS,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Generate SMPL-H motion from text prompt.

    Args:
        prompt: Text description of the motion
        duration: Duration in seconds
        fps: Frames per second
        guidance_scale: CFG scale for generation
        num_inference_steps: Number of diffusion steps
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing motion data and metadata
    """
    global runtime

    if runtime is None:
        load_model()

    # Set random seed if provided
    if seed is not None:
        import torch
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print(f"Generating motion for: '{prompt}'")
    print(f"  Duration: {duration}s, FPS: {fps}, Steps: {num_inference_steps}")
    start_time = time.time()

    # Generate motion using HY-Motion runtime
    motion_result = runtime.generate_motion(
        text=prompt,
        duration=duration,
        cfg_scale=guidance_scale,
        validation_steps=num_inference_steps,
        seed=seed if seed is not None else int(time.time()),
    )

    generation_time = time.time() - start_time
    print(f"Motion generated in {generation_time:.2f}s")

    # Extract motion data - HY-Motion outputs SMPL-H format
    # The output contains motion data with shape (T, 201) where:
    # D = 3 (root trans) + 6 (root orient) + 21*6 (joint rotations) + 22*3 (joint positions)
    motion = motion_result
    if hasattr(motion_result, 'motion'):
        motion = motion_result.motion
    if hasattr(motion, 'cpu'):
        motion = motion.cpu().numpy()
    if not isinstance(motion, np.ndarray):
        motion = np.array(motion)

    num_frames = motion.shape[0] if motion.ndim >= 1 else int(duration * fps)

    motion_data = {
        "motion": motion,
        "fps": fps,
        "duration": duration,
        "num_frames": num_frames,
    }

    # If motion is in the expected 201-dim format, split out components
    if motion.ndim >= 1 and motion.shape[-1] == 201:
        motion_data["root_translation"] = motion[..., :3]
        motion_data["root_orientation"] = motion[..., 3:9]
        motion_data["joint_rotations"] = motion[..., 9:135]
        motion_data["joint_positions"] = motion[..., 135:201]

    return {
        "motion_data": motion_data,
        "metadata": {
            "prompt": prompt,
            "duration": duration,
            "num_frames": num_frames,
            "fps": fps,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "generation_time": generation_time,
        },
    }


def encode_numpy_arrays(data: dict) -> dict:
    """Encode numpy arrays as base64 for JSON serialization."""
    encoded = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            encoded[key] = {
                "data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        elif value is not None:
            encoded[key] = value
    return encoded


def handler(job: dict) -> dict:
    """
    RunPod serverless handler.

    Expected input format:
    {
        "input": {
            "prompt": "person walking forward",
            "duration": 4.0,
            "fps": 30,
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "seed": null
        }
    }
    """
    job_input = job.get("input", {})

    # Validate required fields
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing required field: prompt"}

    # Extract optional parameters with defaults
    params = {
        "prompt": prompt,
        "duration": job_input.get("duration", 4.0),
        "fps": job_input.get("fps", DEFAULT_FPS),
        "guidance_scale": job_input.get("guidance_scale", 7.5),
        "num_inference_steps": job_input.get("num_inference_steps", 50),
        "seed": job_input.get("seed"),
    }

    try:
        result = generate_motion(**params)

        # Encode numpy arrays for JSON transport
        result["motion_data"] = encode_numpy_arrays(result["motion_data"])

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# Load model on cold start
print("Initializing HY-Motion handler...")
try:
    load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Model pre-loading failed: {e}")
    print("Model will be loaded on first request.")

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
