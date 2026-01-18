# HY-Motion-1.0 RunPod Serverless Container
# Optimized for motion generation only (no FBX export)

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV MODEL_PATH=/runpod-volume/models/hy-motion-1.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
# Using --no-cache-dir to reduce memory usage during build
RUN python -m pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install HY-Motion dependencies
# Note: chumpy removed - not needed for inference, only for SMPL fitting
RUN python -m pip install --no-cache-dir \
    numpy>=1.24.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    diffusers>=0.25.0 \
    safetensors>=0.4.0 \
    scipy>=1.10.0 \
    einops>=0.7.0 \
    tqdm>=4.65.0 \
    huggingface_hub>=0.20.0 \
    trimesh

# Install RunPod SDK
RUN python -m pip install --no-cache-dir runpod>=1.6.0

# Create working directory
WORKDIR /app

# Copy handler code
COPY handler.py /app/handler.py
COPY download_model.py /app/download_model.py

# Download model at build time (optional - can also use network volume)
# Uncomment the following if you want the model baked into the image:
# RUN python download_model.py

# Set the entrypoint for RunPod serverless
CMD ["python", "-u", "handler.py"]
