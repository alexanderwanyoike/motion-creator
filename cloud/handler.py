"""
RunPod Serverless Handler for HY-Motion-1.0

Generates SMPL-H motion data from text prompts.
Returns raw motion data (numpy arrays) for local FBX export.
"""

import os
import io
import base64
import time
from typing import Any

import numpy as np
import torch
import runpod

# Configuration
MODEL_REPO = "Tencent-Hunyuan/HY-Motion-1.0"
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/hy-motion-1.0")
DISABLE_PROMPT_ENGINEERING = os.environ.get("DISABLE_PROMPT_ENGINEERING", "False").lower() == "true"
DEFAULT_NUM_FRAMES = 120  # ~4 seconds at 30fps
DEFAULT_FPS = 30

# Global model reference (loaded once, reused across requests)
model = None
device = None


def download_model_if_needed():
    """Download model from HuggingFace if not present."""
    from huggingface_hub import snapshot_download
    import os

    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return MODEL_PATH

    print(f"Downloading {MODEL_REPO} to {MODEL_PATH}...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_PATH,
        local_dir_use_symlinks=False,
    )
    print("Download complete!")
    return MODEL_PATH


def load_model():
    """Load HY-Motion-1.0 model into GPU memory."""
    global model, device

    if model is not None:
        return model

    # Auto-download if needed
    model_path = download_model_if_needed()

    print(f"Loading HY-Motion-1.0 from {model_path}...")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Import HY-Motion pipeline
    # Note: Adjust import path based on actual HY-Motion package structure
    try:
        from hy_motion import HYMotionPipeline
        model = HYMotionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    except ImportError:
        # Fallback: Try loading from diffusers-style pipeline
        from diffusers import DiffusionPipeline
        model = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            custom_pipeline="hy_motion",
        ).to(device)

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")

    return model


def generate_motion(
    prompt: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    fps: int = DEFAULT_FPS,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Generate SMPL-H motion from text prompt.

    Args:
        prompt: Text description of the motion
        num_frames: Number of frames to generate
        fps: Frames per second
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of diffusion steps
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing motion data and metadata
    """
    global model, device

    if model is None:
        load_model()

    # Set random seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    print(f"Generating motion for: '{prompt}'")
    print(f"  Frames: {num_frames}, FPS: {fps}, Steps: {num_inference_steps}")
    start_time = time.time()

    # Generate motion
    with torch.inference_mode():
        output = model(
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

    generation_time = time.time() - start_time
    print(f"Motion generated in {generation_time:.2f}s")

    # Extract motion data from output
    # HY-Motion outputs SMPL-H format: body_pose, global_orient, transl, betas
    motion_data = {
        "body_pose": output.body_pose.cpu().numpy() if hasattr(output, "body_pose") else output.motion.cpu().numpy(),
        "global_orient": output.global_orient.cpu().numpy() if hasattr(output, "global_orient") else None,
        "transl": output.transl.cpu().numpy() if hasattr(output, "transl") else None,
        "betas": output.betas.cpu().numpy() if hasattr(output, "betas") else None,
    }

    # Handle case where output is just a motion tensor
    if not hasattr(output, "body_pose") and hasattr(output, "motion"):
        motion_data["motion"] = output.motion.cpu().numpy()

    return {
        "motion_data": motion_data,
        "metadata": {
            "prompt": prompt,
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
            "prompt": "character walking forward",
            "num_frames": 120,
            "fps": 30,
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "seed": null
        }
    }

    Returns:
    {
        "motion_data": {
            "body_pose": {"data": "base64...", "dtype": "float32", "shape": [120, 63]},
            ...
        },
        "metadata": {...}
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
        "num_frames": job_input.get("num_frames", DEFAULT_NUM_FRAMES),
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


# Load model on cold start (RunPod FlashBoot optimization)
print("Initializing HY-Motion handler...")
load_model()

# Start RunPod serverless
runpod.serverless.start({"handler": handler})
