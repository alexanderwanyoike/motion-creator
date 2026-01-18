# HY-Motion Cloud Pipeline

Generate AI motion animations from text prompts using HY-Motion-1.0 on RunPod serverless, with local FBX export and Mixamo retargeting.

## Architecture

```
┌────────────────────────┐     ┌──────────────────────────────┐
│  Your Machine (Local)  │     │    RunPod Serverless (GPU)   │
│                        │     │                              │
│  1. Send prompt  ─────────▶  │  HY-Motion-1.0 Full          │
│                        │     │  - Text encoding             │
│                        │     │  - Motion generation         │
│  2. Receive motion ◀──────── │  - Output: SMPL-H data       │
│     (.npy/.npz)        │     │    (~100KB)                  │
│                        │     └──────────────────────────────┘
│  3. Local processing:  │
│     - Retarget to      │
│       Mixamo skeleton  │
│     - Export to FBX    │
│     (CPU only, fast)   │
│                        │
│  4. Output: .fbx       │
└────────────────────────┘
```

## Cost Estimate

| Component | Cost |
|-----------|------|
| A100 40GB (20 hrs/month) | ~$35/month |
| Storage | ~$0.80/month |
| **Per-animation** | **~$0.18** |

## Quick Start

### 1. Push to GitHub (Auto-builds Docker Image)

```bash
# Initialize repo and push
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/motion-creator.git
git branch -M main
git push -u origin main
```

GitHub Actions will automatically build and push the Docker image to GitHub Container Registry.

**After the workflow completes**, your image will be at:
```
ghcr.io/YOUR_USERNAME/motion-creator/hy-motion:latest
```

### 2. Make Package Public (Required for RunPod)

RunPod needs to pull your image, so it must be public:

1. Go to your GitHub profile → **Packages**
2. Find `motion-creator/hy-motion`
3. Click **Package settings** → **Change visibility** → **Public**

### 3. Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:
   - **Container Image**: `ghcr.io/YOUR_USERNAME/motion-creator/hy-motion:latest`
   - **GPU**: A100 40GB
   - **Worker Type**: Flex (scale to zero)
   - **Max Workers**: 1-2
   - **Idle Timeout**: 5 seconds
4. Click **Create**
5. Copy your **Endpoint ID** from the dashboard

### 4. Set Up Local Environment

```bash
cd local

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install FBX SDK (required for .fbx export)
# Download from: https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3
# Follow Autodesk's installation instructions for Python bindings

# Configure environment
cp .env.example .env
# Edit .env with your RunPod credentials
```

### 5. Generate Motion

```bash
# Set credentials
export RUNPOD_API_KEY="your-api-key"
export RUNPOD_ENDPOINT_ID="your-endpoint-id"

# Generate animation
python client.py --prompt "character walking forward" --output walk.fbx

# With options
python client.py \
  --prompt "jumping with arms raised" \
  --output jump.fbx \
  --num-frames 90 \
  --fps 30 \
  --seed 42
```

## File Structure

```
motion-creator/
├── .github/
│   └── workflows/
│       └── build-push.yml   # CI/CD: builds & pushes to ghcr.io
│
├── cloud/                    # RunPod serverless (GPU)
│   ├── Dockerfile           # Container with HY-Motion
│   ├── handler.py           # Serverless handler
│   └── download_model.py    # Model downloader
│
├── local/                    # Local pipeline (CPU)
│   ├── requirements.txt     # Python dependencies
│   ├── client.py            # CLI tool
│   ├── retarget.py          # SMPL-H → Mixamo conversion
│   └── export_fbx.py        # FBX file export
│
└── README.md
```

## CLI Reference

```
Usage: client.py [OPTIONS]

Options:
  -p, --prompt TEXT          Text description of motion (required)
  -o, --output PATH          Output file path (.fbx or .npz) (required)
  -n, --num-frames INTEGER   Number of frames (default: 120)
  --fps INTEGER              Frames per second (default: 30)
  -g, --guidance-scale FLOAT Guidance scale (default: 7.5)
  -s, --steps INTEGER        Diffusion steps (default: 50)
  --seed INTEGER             Random seed for reproducibility
  --save-raw / --no-save-raw Also save raw .npz data
  --api-key TEXT             RunPod API key
  --endpoint-id TEXT         RunPod endpoint ID
  --help                     Show this message and exit
```

## Environment Variables

Create a `.env` file in the `local/` directory:

```env
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
```

## FBX SDK Installation

The FBX export requires Autodesk's FBX SDK with Python bindings:

1. Download FBX SDK from [Autodesk Developer Network](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3)
2. Install the SDK following Autodesk's instructions
3. Install Python bindings (usually in the SDK's `lib/Python` directory)

If you can't install FBX SDK, use `.npz` output format instead:

```bash
python client.py --prompt "walking" --output walk.npz
```

## Using with flip-frenzy

The exported FBX files use standard Mixamo skeleton naming, compatible with flip-frenzy characters:

1. Generate animation: `python client.py -p "flip trick" -o flip.fbx`
2. Import `flip.fbx` into your game engine
3. Apply to any Mixamo-rigged character
4. The skeleton mapping should work automatically

## Troubleshooting

### "FBX SDK not available"
Install FBX SDK from Autodesk, or use `.npz` output format.

### "RUNPOD_API_KEY not set"
Set your RunPod API key:
```bash
export RUNPOD_API_KEY="your-key"
```

### "Job timed out"
The default timeout is 10 minutes. For longer generations, the RunPod worker may need more time to cold start. Try again or check RunPod dashboard.

### Motion looks wrong
- Check that your character uses standard Mixamo skeleton naming
- Try adjusting `--guidance-scale` for different motion styles
- Use `--save-raw` to inspect the raw SMPL-H data

## RunPod Setup Guide

Complete walkthrough for deploying to RunPod serverless.

### Step 1: Create RunPod Account

1. Go to [runpod.io](https://www.runpod.io)
2. Click **Sign Up** (top right)
3. Create account with email or GitHub

### Step 2: Add Credits

1. Go to [Billing](https://www.runpod.io/console/user/billing)
2. Add payment method
3. Add credits ($10-25 is enough to start testing)

### Step 3: Get Your API Key

1. Go to [Settings → API Keys](https://www.runpod.io/console/user/settings)
2. Click **Create API Key**
3. Give it a name like "motion-creator"
4. **Copy and save the key** - you won't see it again

```bash
export RUNPOD_API_KEY="rp_xxxxxxxxxxxxxxxx"
```

### Step 4: Create Network Volume (Model Storage)

The HY-Motion model is ~8GB. Use a network volume instead of baking it into Docker:

1. Go to [Storage → Network Volumes](https://www.runpod.io/console/user/storage)
2. Click **+ New Network Volume**
3. Configure:
   - **Name:** `hy-motion-models`
   - **Region:** Pick one close to you (e.g., `US-TX-3`)
   - **Size:** 20 GB
4. Click **Create**
5. **Note the region** - your endpoint must be in the same region

### Step 5: Create Serverless Endpoint

1. Go to [Serverless → Endpoints](https://www.runpod.io/console/serverless)
2. Click **+ New Endpoint**
3. Fill in:

| Field | Value |
|-------|-------|
| **Endpoint Name** | `hy-motion` |
| **Container Image** | `ghcr.io/YOUR_USERNAME/motion-creator/hy-motion:latest` |
| **Container Disk** | 20 GB |

4. Under **GPU Configuration**:
   - Select **A100 40GB** (or A100 80GB if unavailable)

5. Under **Worker Configuration**:
   - **Max Workers:** 1 (start small)
   - **Idle Timeout:** 5 seconds
   - **Execution Timeout:** 600 seconds

6. Under **Advanced**:
   - **Select Network Volume:** Choose `hy-motion-models`
   - **Volume Mount Path:** `/runpod-volume`

7. Click **Create Endpoint**

### Step 6: Copy Your Endpoint ID

After creation, you'll see your endpoint in the list. Copy the **Endpoint ID** (looks like `abc123xyz`).

```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
```

### Step 7: Test Your Endpoint

Test with curl:

```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "person walking forward"}}'

# Check status (replace JOB_ID)
curl "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/JOB_ID" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

### Recommended Settings Summary

| Setting | Value |
|---------|-------|
| GPU | A100 40GB |
| Worker Type | Flex (scale to zero) |
| Max Workers | 1-2 (personal use) |
| Idle Timeout | 5 seconds |
| Execution Timeout | 600 seconds |
| Network Volume | 20 GB, same region as endpoint |

## License

MIT License - see individual dependencies for their licenses.
