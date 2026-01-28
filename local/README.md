# Motion Creator - Local Client

Local utilities for Motion Creator including the Gradio web UI and CLI client.

> See the [main README](../README.md) for full setup instructions (RunPod account, API keys, endpoint creation).

## Quick Start

After completing setup from the main README:

```bash
cd local
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Then edit with your RunPod credentials
```

## Usage

### Web UI (Gradio)

```bash
python app.py
```

Open http://127.0.0.1:7860 and enter a motion prompt.

### CLI

```bash
python client.py -p "person walking forward" -o walk.npz
```

See `python client.py --help` for all options.

## Output

Generated motion is saved to `output/gradio/` (web UI) or the path you specify (CLI):

- `.npz` - Motion data (Rh, poses, trans, betas)
- `_meta.json` - Metadata (prompt, settings)
