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
HY_MOTION_DIR = "/app/hy-motion"
sys.path.insert(0, HY_MOTION_DIR)

# Configuration
VOLUME_PATH = os.environ.get("VOLUME_PATH", "/runpod-volume")
USE_LITE_MODEL = os.environ.get("USE_LITE_MODEL", "false").lower() == "true"
DEFAULT_FPS = 30

# Model info
MODEL_NAME = "HY-Motion-1.0-Lite" if USE_LITE_MODEL else "HY-Motion-1.0"

# Global runtime reference
runtime = None


def setup_model_symlinks():
    """Create symlinks from ckpts/ to network volume where models are stored."""
    ckpts_dir = os.path.join(HY_MOTION_DIR, "ckpts")
    volume_ckpts = os.path.join(VOLUME_PATH, "ckpts")

    # Create volume ckpts dir if needed
    os.makedirs(volume_ckpts, exist_ok=True)

    # Remove existing ckpts dir if it's not a symlink
    if os.path.exists(ckpts_dir) and not os.path.islink(ckpts_dir):
        import shutil
        shutil.rmtree(ckpts_dir)

    # Create symlink
    if not os.path.exists(ckpts_dir):
        os.symlink(volume_ckpts, ckpts_dir)
        print(f"Created symlink: {ckpts_dir} -> {volume_ckpts}")


def download_hy_motion_checkpoint():
    """Download HY-Motion checkpoint to network volume if not present.

    Note: CLIP and Qwen3-8B text encoders are downloaded automatically by
    HY-Motion using HuggingFace Hub (USE_HF_MODELS=1 is default).
    They get cached to HF_HOME which points to the network volume.
    """
    from huggingface_hub import snapshot_download

    ckpts_dir = os.path.join(VOLUME_PATH, "ckpts")
    model_path = os.path.join(ckpts_dir, "tencent", MODEL_NAME)

    # Check if HY-Motion checkpoint exists
    ckpt_file = os.path.join(model_path, "latest.ckpt")
    if os.path.exists(ckpt_file):
        print(f"HY-Motion checkpoint already exists: {model_path}")
        return

    print(f"Downloading tencent/HY-Motion-1.0 ({MODEL_NAME})...")
    os.makedirs(os.path.join(ckpts_dir, "tencent"), exist_ok=True)

    snapshot_download(
        repo_id="tencent/HY-Motion-1.0",
        local_dir=os.path.join(ckpts_dir, "tencent"),
        local_dir_use_symlinks=False,
        allow_patterns=[f"{MODEL_NAME}/*"],
    )
    print(f"Downloaded: {MODEL_NAME}")


def load_model():
    """Load HY-Motion-1.0 runtime."""
    global runtime

    if runtime is not None:
        return runtime

    # Set up symlinks and download HY-Motion checkpoint
    setup_model_symlinks()
    download_hy_motion_checkpoint()

    print(f"Loading {MODEL_NAME}...")
    start_time = time.time()

    # Change to hy-motion directory so relative paths work
    os.chdir(HY_MOTION_DIR)

    # DEBUG: Check if stats files exist
    stats_dir = os.path.join(HY_MOTION_DIR, "stats")
    print(f">>> DEBUG: Checking stats directory: {stats_dir}")
    print(f">>> DEBUG: stats dir exists: {os.path.exists(stats_dir)}")
    if os.path.exists(stats_dir):
        print(f">>> DEBUG: stats contents: {os.listdir(stats_dir)}")
        mean_path = os.path.join(stats_dir, "Mean.npy")
        std_path = os.path.join(stats_dir, "Std.npy")
        print(f">>> DEBUG: Mean.npy exists: {os.path.exists(mean_path)}")
        print(f">>> DEBUG: Std.npy exists: {os.path.exists(std_path)}")
        if os.path.exists(mean_path):
            mean = np.load(mean_path)
            print(f">>> DEBUG: Mean shape: {mean.shape}, first 5 values: {mean[:5]}")
        if os.path.exists(std_path):
            std = np.load(std_path)
            print(f">>> DEBUG: Std shape: {std.shape}, first 5 values: {std[:5]}")

    # Import HY-Motion runtime
    from hymotion.utils.t2m_runtime import T2MRuntime

    # Model path relative to hy-motion dir
    model_path = f"ckpts/tencent/{MODEL_NAME}"
    config_path = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")

    print(f">>> DEBUG: config_path: {config_path}, exists: {os.path.exists(config_path)}")
    print(f">>> DEBUG: ckpt_path: {ckpt_path}, exists: {os.path.exists(ckpt_path)}")

    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_path,
        device_ids=[0],
        disable_prompt_engineering=True,
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
    """
    global runtime

    if runtime is None:
        load_model()

    # Set random seed if provided
    if seed is None:
        seed = int(time.time()) % 2**32

    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Generating motion for: '{prompt}'")
    print(f"  Duration: {duration}s, FPS: {fps}, Seed: {seed}")
    start_time = time.time()

    # Generate motion using HY-Motion runtime
    # API: text, seeds_csv, duration, cfg_scale, output_format, ...
    # Valid formats: "fbx" or "dict"
    html_content, fbx_paths, model_output = runtime.generate_motion(
        text=prompt,
        seeds_csv=str(seed),
        duration=duration,
        cfg_scale=guidance_scale,
        output_format="dict",  # Get raw motion data as dict
    )

    generation_time = time.time() - start_time
    print(f"Motion generated in {generation_time:.2f}s")

    # model_output is a dict with keys:
    # - latent_denorm: (B, L, 201) raw motion
    # - keypoints3d: (B, L, J, 3) 3D joint positions
    # - rot6d: (B, L, J, 6) 6D rotations
    # - transl: (B, L, 3) root translation
    # - root_rotations_mat: (B, L, 3, 3) root rotation matrices
    # - text: input prompt
    print(f"model_output type: {type(model_output)}")
    if isinstance(model_output, dict):
        print(f"model_output keys: {list(model_output.keys())}")

    # Helper to convert tensors to numpy
    def to_numpy(x):
        if hasattr(x, 'cpu'):
            x = x.cpu()
        if hasattr(x, 'numpy'):
            x = x.numpy()
        return np.array(x) if not isinstance(x, np.ndarray) else x

    # Extract motion data from model_output dict
    motion_data = {
        "fps": fps,
        "duration": duration,
    }

    if isinstance(model_output, dict):
        # Get the 201-dim latent motion representation
        if "latent_denorm" in model_output:
            latent = to_numpy(model_output["latent_denorm"])
            # Remove batch dimension if present: (B, L, 201) -> (L, 201)
            if latent.ndim == 3 and latent.shape[0] == 1:
                latent = latent[0]
            motion_data["motion"] = latent
            motion_data["num_frames"] = latent.shape[0]

            # Split 201-dim into components
            if latent.shape[-1] == 201:
                motion_data["root_translation"] = latent[..., :3]
                motion_data["root_orientation"] = latent[..., 3:9]
                motion_data["joint_rotations"] = latent[..., 9:135]
                motion_data["joint_positions"] = latent[..., 135:201]

        # Also include other useful data
        if "keypoints3d" in model_output:
            kp = to_numpy(model_output["keypoints3d"])
            if kp.ndim == 4 and kp.shape[0] == 1:
                kp = kp[0]
            motion_data["keypoints3d"] = kp

        if "rot6d" in model_output:
            rot = to_numpy(model_output["rot6d"])
            if rot.ndim == 4 and rot.shape[0] == 1:
                rot = rot[0]
            motion_data["rot6d"] = rot

        if "transl" in model_output:
            transl = to_numpy(model_output["transl"])
            if transl.ndim == 3 and transl.shape[0] == 1:
                transl = transl[0]
            motion_data["transl"] = transl
    else:
        # Fallback for unexpected format
        motion_data["motion"] = to_numpy(model_output)
        motion_data["num_frames"] = int(duration * fps)

    return {
        "motion_data": motion_data,
        "metadata": {
            "prompt": prompt,
            "duration": duration,
            "num_frames": motion_data.get("num_frames", int(duration * fps)),
            "fps": fps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "generation_time": generation_time,
            "model": MODEL_NAME,
        },
    }


def encode_numpy_arrays(data: dict) -> dict:
    """Encode numpy arrays as base64 for JSON serialization."""
    encoded = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # Ensure array is a numeric type (convert object arrays)
            if value.dtype == object:
                try:
                    value = np.array(value, dtype=np.float32)
                except (ValueError, TypeError):
                    # Skip arrays that can't be converted
                    print(f"Warning: Skipping {key} - cannot convert to float array")
                    continue
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
    """RunPod serverless handler."""
    job_input = job.get("input", {})

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "Missing required field: prompt"}

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
