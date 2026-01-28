#!/usr/bin/env python3
"""
HY-Motion Gradio Client

Uses RunPod for inference and HY-Motion's proper mesh visualization.
"""

import os
import io
import json
import base64
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import requests
import gradio as gr
from dotenv import load_dotenv

# Import HY-Motion visualization utilities
from hymotion.pipeline.body_model import construct_smpl_data_dict
from hymotion.utils.visualize_mesh_web import generate_static_html_content

load_dotenv()

# RunPod config
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
RUNPOD_API_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# Output directory for NPZ files
OUTPUT_DIR = Path(__file__).parent / "output" / "gradio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def decode_numpy(encoded: dict) -> np.ndarray:
    """Decode base64 numpy array from RunPod response."""
    data = base64.b64decode(encoded["data"])
    return np.load(io.BytesIO(data), allow_pickle=False)


def call_runpod(prompt: str, duration: float, seed: int | None = None) -> dict:
    """Call RunPod endpoint for motion generation."""
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": {
            "prompt": prompt,
            "duration": duration,
            "fps": 30,
        }
    }
    if seed is not None:
        payload["input"]["seed"] = seed

    # Submit job (synchronous)
    response = requests.post(
        f"{RUNPOD_API_URL}/runsync",
        headers=headers,
        json=payload,
        timeout=300
    )
    response.raise_for_status()
    result = response.json()

    if result.get("status") == "FAILED":
        raise RuntimeError(f"RunPod job failed: {result.get('error')}")

    return result.get("output", {})


def save_motion_data(rot6d: np.ndarray, transl: np.ndarray, prompt: str, timestamp: str) -> str:
    """
    Save motion data in HY-Motion's NPZ format.

    Args:
        rot6d: Rotation data in 6D format, shape (frames, joints, 6)
        transl: Translation data, shape (frames, 3)
        prompt: The text prompt
        timestamp: Unique timestamp for filename

    Returns:
        base_filename: The base filename (without extension) for later retrieval
    """
    # Convert numpy to torch for construct_smpl_data_dict
    rot6d_tensor = torch.from_numpy(rot6d).float()
    transl_tensor = torch.from_numpy(transl).float()

    # Use HY-Motion's construct_smpl_data_dict to convert rot6d to SMPL format
    smpl_data = construct_smpl_data_dict(rot6d_tensor, transl_tensor)

    # Save metadata JSON
    base_filename = timestamp
    meta_path = OUTPUT_DIR / f"{base_filename}_meta.json"
    meta_data = {
        "timestamp": timestamp,
        "text": prompt,
        "text_rewrite": [prompt],
        "num_samples": 1,
        "base_filename": base_filename,
    }
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)

    # Prepare NPZ data
    npz_dict = {
        "gender": np.array([smpl_data.get("gender", "neutral")], dtype=str),
    }

    for key in ["Rh", "trans", "poses", "betas"]:
        if key in smpl_data:
            val = smpl_data[key]
            if isinstance(val, (list, tuple)):
                val = np.array(val)
            elif isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            npz_dict[key] = val

    # Save NPZ file (sample index 000)
    sample_path = OUTPUT_DIR / f"{base_filename}_000.npz"
    np.savez_compressed(sample_path, **npz_dict)

    print(f"Saved motion data to {sample_path}")
    print(f"  - Rh shape: {npz_dict.get('Rh', []).shape if 'Rh' in npz_dict else 'N/A'}")
    print(f"  - trans shape: {npz_dict.get('trans', []).shape if 'trans' in npz_dict else 'N/A'}")
    print(f"  - poses shape: {npz_dict.get('poses', []).shape if 'poses' in npz_dict else 'N/A'}")

    return base_filename


def generate_motion(prompt: str, duration: float, seed: int | None = None):
    """Generate motion and return HTML visualization using HY-Motion's renderer."""
    if not prompt:
        return "<p>Please enter a prompt</p>", None

    try:
        # Call RunPod
        print(f"Calling RunPod with prompt: '{prompt}', duration: {duration}")
        output = call_runpod(prompt, duration, seed if seed and seed > 0 else None)

        motion_data = output.get("motion_data", {})
        metadata = output.get("metadata", {})

        # Get rot6d and transl from response
        if "rot6d" not in motion_data:
            return "<p>No rot6d data in response</p>", metadata

        rot6d = decode_numpy(motion_data["rot6d"])
        transl = decode_numpy(motion_data["transl"])

        print(f"Received rot6d shape: {rot6d.shape}, transl shape: {transl.shape}")

        # Generate unique timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save motion data in HY-Motion's format
        base_filename = save_motion_data(rot6d, transl, prompt, timestamp)

        # Generate HTML using HY-Motion's visualization
        # The folder_name is relative to the output base directory
        html_content = generate_static_html_content(
            folder_name="output/gradio",
            file_name=base_filename,
            hide_captions=False,
        )

        # Wrap in iframe for Gradio display
        # Escape single quotes for srcdoc
        escaped_html = html_content.replace("'", "&#39;")
        iframe_html = f"<iframe srcdoc='{escaped_html}' width='100%' height='700' frameborder='0' style='border-radius: 8px;'></iframe>"

        return iframe_html, metadata

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<p style='color:red'>Error: {e}</p>", None


def create_app():
    """Create Gradio app."""
    with gr.Blocks(title="HY-Motion Client") as app:
        gr.Markdown("# HY-Motion Client\nGenerate motion from text using RunPod + HY-Motion visualization")

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a person walking forward",
                    lines=2,
                )
                duration = gr.Slider(
                    1, 10, value=4, step=0.5,
                    label="Duration (seconds)"
                )
                seed = gr.Number(
                    label="Seed (optional, 0 = random)",
                    value=0,
                    precision=0
                )
                generate_btn = gr.Button("Generate Motion", variant="primary")

                metadata_output = gr.JSON(label="Response Metadata")

            with gr.Column(scale=2):
                html_output = gr.HTML(label="Motion Visualization")

        generate_btn.click(
            generate_motion,
            inputs=[prompt, duration, seed],
            outputs=[html_output, metadata_output]
        )

        gr.Markdown("""
        ### Instructions
        - Enter a motion description (e.g., "a person walking forward", "a person jumping")
        - Set the duration and optionally a seed for reproducibility
        - Click Generate Motion and wait for the result
        - Use mouse to rotate/zoom the 3D view, click to play/pause
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
