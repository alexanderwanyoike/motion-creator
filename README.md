# Motion Creator

Generate AI motion animations from text prompts using [HY-Motion-1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) on RunPod serverless. Includes a Gradio web UI for interactive generation and a CLI for scripted workflows, with local FBX export and Mixamo retargeting.

## Architecture

```
┌────────────────────────┐     ┌──────────────────────────────┐
│  Your Machine (Local)  │     │    RunPod Serverless (GPU)   │
│                        │     │                              │
│  1. Send prompt  ─────────▶  │  HY-Motion-1.0 (1B params)   │
│                        │     │  - Text → Motion generation  │
│  2. Receive motion ◀──────── │  - Output: SMPL-H data       │
│     (base64 numpy)     │     │    (~100KB)                  │
│                        │     └──────────────────────────────┘
│  3. Local processing:  │
│     - Retarget to      │
│       Mixamo skeleton  │
│     - Export to FBX    │
│                        │
│  4. Output: .fbx       │
└────────────────────────┘
```

## Cost Estimate

| Component | Cost |
|-----------|------|
| 48GB GPU (20 hrs/month) | ~$25-40/month |
| Network Volume (20GB) | ~$2/month |
| **Per-animation** | **~$0.15-0.20** |

---

## Setup Guide

### Step 1: Push to GitHub

The Docker image builds automatically via GitHub Actions:

```bash
git clone https://github.com/YOUR_USERNAME/motion-creator.git
cd motion-creator
git push origin main
```

Image will be at: `ghcr.io/YOUR_USERNAME/motion-creator/hy-motion:latest`

**Make the package public** (required for RunPod to pull it):
1. Go to github.com → Your Profile → Packages
2. Click `motion-creator/hy-motion`
3. Package Settings → Change visibility → **Public**

### Step 2: Create RunPod Account & Add Credits

1. Sign up at [runpod.io](https://www.runpod.io)
2. Go to [Billing](https://www.runpod.io/console/user/billing) → Add $10-25 credits

### Step 3: Get Your API Key

1. Go to [Settings → API Keys](https://www.runpod.io/console/user/settings)
2. Click **Create API Key**
3. Save it somewhere safe

```bash
export RUNPOD_API_KEY="rp_xxxxxxxxxxxxxxxx"
```

### Step 4: Create Network Volume

The model weights (~8GB) are stored here and persist between cold starts:

1. Go to [Storage → Network Volumes](https://www.runpod.io/console/user/storage)
2. Click **+ New Network Volume**
3. Configure:
   - **Name:** `hy-motion-models`
   - **Region:** Pick one with good GPU availability (check Step 5 first)
   - **Size:** 20 GB
4. **Remember the region** - your endpoint must be in the same region

### Step 5: Create Serverless Endpoint

1. Go to [Serverless → Endpoints](https://www.runpod.io/console/serverless)
2. Click **+ New Endpoint**
3. Scroll down to **Container Image** and enter:
   ```
   ghcr.io/YOUR_USERNAME/motion-creator/hy-motion:latest
   ```
4. Select **48 GB GPU** (HY-Motion needs ~26GB VRAM)
5. Configure workers:
   - **Max Workers:** 1
   - **Idle Timeout:** 5 seconds
   - **Execution Timeout:** 600 seconds
6. Under **Advanced** → **Network Volume**:
   - Select your `hy-motion-models` volume
   - Mount path: `/runpod-volume`
7. Click **Create**
8. Copy your **Endpoint ID**

```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
```

### Step 6: First Run (Downloads Model)

The first request downloads the model to your network volume (~5-10 min):

```bash
cd local
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export RUNPOD_API_KEY="your-key"
export RUNPOD_ENDPOINT_ID="your-endpoint-id"

# First run - will download model
python client.py -p "person walking forward" -o walk.npz
```

Watch the logs in RunPod dashboard to see download progress.

### Step 7: Generate Animations

After the model is cached, generation is fast (~30-60s cold start, ~5-10s generation):

```bash
# Basic usage
python client.py -p "character doing a backflip" -o backflip.npz

# With options
python client.py \
  -p "person running and then jumping" \
  -o run_jump.npz \
  --duration 5.0 \
  --fps 30 \
  --seed 42
```

---

## CLI Reference

```
Usage: client.py [OPTIONS]

Options:
  -p, --prompt TEXT          Motion description (required)
  -o, --output PATH          Output file (.fbx or .npz) (required)
  --duration FLOAT           Duration in seconds (default: 4.0)
  --fps INTEGER              Frames per second (default: 30)
  -g, --guidance-scale FLOAT CFG scale (default: 7.5)
  -s, --steps INTEGER        Diffusion steps (default: 50)
  --seed INTEGER             Random seed for reproducibility
  --save-raw                 Also save raw motion data
  --help                     Show this message
```

---

## Web UI (Gradio)

For interactive use, run the Gradio web app:

```bash
cd local
python app.py
```

Open http://127.0.0.1:7860 in your browser.

1. Enter a motion description (e.g., "a person walking forward")
2. Set duration (1-10 seconds)
3. Optionally set a seed for reproducibility
4. Click "Generate Motion"
5. View the animated 3D visualization

Output is saved to `output/gradio/` as `.npz` motion data and metadata JSON.

---

## File Structure

```
motion-creator/
├── .github/workflows/
│   └── build-push.yml      # CI/CD: builds & pushes to ghcr.io
├── cloud/
│   ├── Dockerfile          # Clones HY-Motion, installs deps
│   ├── handler.py          # RunPod serverless handler
│   └── stats/
│       ├── Mean.npy        # Motion normalization stats
│       └── Std.npy
├── local/
│   ├── app.py              # Gradio web UI
│   ├── client.py           # CLI tool
│   ├── retarget.py         # SMPL-H → Mixamo conversion
│   ├── export_fbx.py       # FBX export (requires FBX SDK)
│   ├── visualize.py        # Motion visualization utilities
│   ├── hymotion/           # Local HY-Motion utilities
│   │   ├── pipeline/       # Body model handling
│   │   └── utils/          # Geometry & web visualization
│   ├── scripts/
│   │   └── gradio/         # Web templates & assets
│   ├── requirements.txt
│   └── .env.example
└── README.md
```

---

## Troubleshooting

### "Model pre-loading failed"
Normal on first run. The model downloads from HuggingFace to your network volume (~5-10 min). Check RunPod logs.

### "No module named 'hymotion'"
The Docker image didn't build correctly. Check GitHub Actions logs.

### Job times out
- First run can take 10+ minutes (model download)
- Increase execution timeout in endpoint settings

### Wrong region for network volume
Network volumes are region-locked. Delete and recreate in a region with 48GB GPU availability.

### GPU not available
48GB GPUs can have limited availability. Try:
- Different region
- Wait and retry
- Use 80GB GPU (more expensive but more available)

---

## FBX Export (Optional)

To export `.fbx` files, install Autodesk FBX SDK:

1. Download from [Autodesk](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3)
2. Install Python bindings
3. Then use: `python client.py -p "walking" -o walk.fbx`

Without FBX SDK, use `.npz` format and convert in Blender.

---

## Links

- [HY-Motion-1.0 GitHub](https://github.com/Tencent-Hunyuan/HY-Motion-1.0)
- [HY-Motion HuggingFace](https://huggingface.co/tencent/HY-Motion-1.0)
- [RunPod Serverless Docs](https://docs.runpod.io/serverless)
