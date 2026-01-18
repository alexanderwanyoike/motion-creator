#!/usr/bin/env python3
"""
HY-Motion Client CLI

Generates motion from text prompts using RunPod serverless,
then exports to FBX with Mixamo retargeting.

Usage:
    python client.py --prompt "character walking forward" --output walk.fbx
    python client.py --prompt "jumping" --output jump.fbx --fps 60
"""

import os
import io
import sys
import base64
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from retarget import retarget_from_motion_dict
from export_fbx import export_motion_to_fbx, check_fbx_available

# Load environment variables
load_dotenv()

console = Console()

# Configuration
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
RUNPOD_API_URL = "https://api.runpod.ai/v2"

# Polling configuration
POLL_INTERVAL = 2.0  # seconds
MAX_POLL_TIME = 600  # 10 minutes timeout


def decode_numpy_array(encoded: dict) -> np.ndarray:
    """Decode base64-encoded numpy array."""
    data = base64.b64decode(encoded["data"])
    buffer = io.BytesIO(data)
    return np.load(buffer, allow_pickle=False)


def decode_motion_data(encoded_data: dict) -> dict:
    """Decode all numpy arrays in motion data."""
    decoded = {}
    for key, value in encoded_data.items():
        if isinstance(value, dict) and "data" in value:
            decoded[key] = decode_numpy_array(value)
        elif value is not None:
            decoded[key] = value
    return decoded


class RunPodClient:
    """Client for RunPod serverless API."""

    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"{RUNPOD_API_URL}/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def submit_job(self, prompt: str, **kwargs) -> str:
        """
        Submit a motion generation job.

        Returns:
            Job ID for status polling
        """
        payload = {
            "input": {
                "prompt": prompt,
                **kwargs,
            }
        }

        response = requests.post(
            f"{self.base_url}/run",
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        return result["id"]

    def get_job_status(self, job_id: str) -> dict:
        """Get status of a submitted job."""
        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = POLL_INTERVAL,
        max_time: float = MAX_POLL_TIME,
    ) -> dict:
        """
        Wait for job completion with polling.

        Returns:
            Job result including motion data
        """
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating motion...", total=None)

            while True:
                elapsed = time.time() - start_time
                if elapsed > max_time:
                    raise TimeoutError(f"Job timed out after {max_time}s")

                status = self.get_job_status(job_id)
                job_status = status.get("status")

                progress.update(
                    task,
                    description=f"Generating motion... ({job_status}, {elapsed:.0f}s)",
                )

                if job_status == "COMPLETED":
                    return status.get("output", {})
                elif job_status == "FAILED":
                    error = status.get("error", "Unknown error")
                    raise RuntimeError(f"Job failed: {error}")
                elif job_status in ("IN_QUEUE", "IN_PROGRESS"):
                    time.sleep(poll_interval)
                else:
                    console.print(f"[yellow]Unknown status: {job_status}[/yellow]")
                    time.sleep(poll_interval)

    def generate_motion(self, prompt: str, **kwargs) -> dict:
        """
        Generate motion from prompt (blocking).

        Returns:
            Decoded motion data dictionary
        """
        job_id = self.submit_job(prompt, **kwargs)
        console.print(f"[dim]Job ID: {job_id}[/dim]")

        result = self.wait_for_completion(job_id)

        if "error" in result:
            raise RuntimeError(f"Generation failed: {result['error']}")

        # Decode motion data
        motion_data = result.get("motion_data", {})
        return decode_motion_data(motion_data), result.get("metadata", {})


def save_motion_npz(motion_data: dict, output_path: Path):
    """Save raw motion data as NPZ file."""
    np.savez(output_path, **motion_data)


@click.command()
@click.option(
    "--prompt", "-p",
    required=True,
    help="Text description of the motion to generate",
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output file path (.fbx or .npz)",
)
@click.option(
    "--duration", "-d",
    default=4.0,
    help="Duration in seconds (default: 4.0)",
)
@click.option(
    "--fps",
    default=30,
    help="Frames per second (default: 30)",
)
@click.option(
    "--guidance-scale", "-g",
    default=7.5,
    help="Classifier-free guidance scale (default: 7.5)",
)
@click.option(
    "--steps", "-s",
    default=50,
    help="Number of diffusion steps (default: 50)",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--save-raw/--no-save-raw",
    default=False,
    help="Also save raw motion data as .npz",
)
@click.option(
    "--api-key",
    envvar="RUNPOD_API_KEY",
    help="RunPod API key (or set RUNPOD_API_KEY env var)",
)
@click.option(
    "--endpoint-id",
    envvar="RUNPOD_ENDPOINT_ID",
    help="RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID env var)",
)
def main(
    prompt: str,
    output: str,
    duration: float,
    fps: int,
    guidance_scale: float,
    steps: int,
    seed: Optional[int],
    save_raw: bool,
    api_key: Optional[str],
    endpoint_id: Optional[str],
):
    """Generate motion from text prompt and export to FBX."""
    output_path = Path(output)

    # Validate configuration
    if not api_key:
        console.print("[red]Error: RUNPOD_API_KEY not set[/red]")
        console.print("Set it via --api-key or RUNPOD_API_KEY environment variable")
        sys.exit(1)

    if not endpoint_id:
        console.print("[red]Error: RUNPOD_ENDPOINT_ID not set[/red]")
        console.print("Set it via --endpoint-id or RUNPOD_ENDPOINT_ID environment variable")
        sys.exit(1)

    # Check FBX SDK for .fbx output
    if output_path.suffix.lower() == ".fbx" and not check_fbx_available():
        console.print("[red]Error: FBX SDK not available for .fbx export[/red]")
        console.print("Install FBX SDK or use .npz output format")
        sys.exit(1)

    console.print(f"[bold]HY-Motion Generator[/bold]")
    console.print(f"  Prompt: {prompt}")
    console.print(f"  Duration: {duration}s @ {fps} fps")
    console.print(f"  Output: {output_path}")
    console.print()

    # Initialize client
    client = RunPodClient(api_key, endpoint_id)

    try:
        # Generate motion
        console.print("[bold blue]Step 1/3:[/bold blue] Generating motion on RunPod...")
        motion_data, metadata = client.generate_motion(
            prompt=prompt,
            duration=duration,
            fps=fps,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            seed=seed,
        )

        gen_time = metadata.get("generation_time", 0)
        console.print(f"[green]Motion generated in {gen_time:.1f}s[/green]")

        # Save raw data if requested
        if save_raw:
            raw_path = output_path.with_suffix(".npz")
            save_motion_npz(motion_data, raw_path)
            console.print(f"[dim]Raw data saved to {raw_path}[/dim]")

        # Retarget to Mixamo
        console.print("[bold blue]Step 2/3:[/bold blue] Retargeting to Mixamo skeleton...")
        retargeted = retarget_from_motion_dict(motion_data, fps=fps)
        console.print("[green]Retargeting complete[/green]")

        # Export to FBX
        console.print("[bold blue]Step 3/3:[/bold blue] Exporting to FBX...")

        if output_path.suffix.lower() == ".fbx":
            success = export_motion_to_fbx(retargeted, output_path, anim_name=output_path.stem)
            if success:
                console.print(f"[green bold]Exported to {output_path}[/green bold]")
            else:
                console.print("[red]FBX export failed[/red]")
                sys.exit(1)
        elif output_path.suffix.lower() == ".npz":
            # Just save the retargeted data
            np.savez(
                output_path,
                joint_rotations={k: v for k, v in retargeted.joint_rotations.items()},
                root_positions=retargeted.root_positions,
                fps=retargeted.fps,
            )
            console.print(f"[green bold]Saved to {output_path}[/green bold]")
        else:
            console.print(f"[red]Unsupported output format: {output_path.suffix}[/red]")
            sys.exit(1)

        console.print()
        console.print("[bold green]Done![/bold green]")

    except requests.exceptions.RequestException as e:
        console.print(f"[red]API Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
