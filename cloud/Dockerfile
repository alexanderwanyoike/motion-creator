# HY-Motion-1.0 RunPod Serverless Container
# Properly installs HY-Motion from official repo

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Initialize git-lfs
RUN git lfs install

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support FIRST
RUN python -m pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Clone HY-Motion-1.0 repository
WORKDIR /app
RUN git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git /app/hy-motion

# Install HY-Motion requirements
WORKDIR /app/hy-motion
RUN python -m pip install --no-cache-dir -r requirements.txt

# Install RunPod SDK
RUN python -m pip install --no-cache-dir runpod>=1.6.0

# Copy our custom handler
COPY handler.py /app/handler.py

# Set working directory back to /app
WORKDIR /app

# Set the entrypoint for RunPod serverless
CMD ["python", "-u", "handler.py"]
