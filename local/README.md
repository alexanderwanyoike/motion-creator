# HY-Motion Local Client

A Gradio-based client for generating motion from text prompts using HY-Motion running on RunPod serverless.

## Features

- Text-to-motion generation using Tencent's HY-Motion-1.0 model
- Real-time 3D visualization using HY-Motion's wooden mesh renderer
- Adjustable duration and seed for reproducible results
- Exports motion data in NPZ format compatible with HY-Motion

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your RunPod credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
```

## Usage

Start the Gradio app:
```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser.

1. Enter a motion description (e.g., "a person walking forward")
2. Set duration (1-10 seconds)
3. Optionally set a seed for reproducibility
4. Click "Generate Motion"
5. View the animated 3D visualization

## Output

Generated motion data is saved to `output/gradio/` as:
- `{timestamp}_000.npz` - Motion data (Rh, poses, trans, betas)
- `{timestamp}_meta.json` - Metadata (prompt, settings)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Gradio Client  │────▶│  RunPod Server   │────▶│   HY-Motion     │
│   (app.py)      │◀────│  (handler.py)    │◀────│   Model GPU     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐
│  3D Visualization│
│  (Three.js mesh) │
└─────────────────┘
```

## Files

- `app.py` - Main Gradio application
- `hymotion/` - HY-Motion utilities for data conversion and visualization
- `scripts/gradio/templates/` - HTML templates for 3D rendering

## Requirements

- Python 3.10+
- RunPod account with HY-Motion endpoint deployed
- ~500MB disk space for dependencies

## Credits

- [HY-Motion](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) by Tencent
- RunPod for serverless GPU infrastructure
