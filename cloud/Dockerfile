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
    python3.10-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install HY-Motion dependencies
RUN pip install \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    diffusers>=0.25.0 \
    safetensors>=0.4.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    einops>=0.7.0 \
    tqdm>=4.65.0 \
    huggingface_hub>=0.20.0 \
    smplx>=0.1.28 \
    chumpy \
    trimesh

# Install RunPod SDK
RUN pip install runpod>=1.6.0

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
