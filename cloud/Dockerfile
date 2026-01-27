# HY-Motion-1.0 RunPod Serverless Container
# Uses official RunPod PyTorch image for guaranteed GPU compatibility

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV USE_HF_MODELS=1

# Install git-lfs (git, python, pytorch already in base image)
RUN apt-get update && apt-get install -y git-lfs ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Clone HY-Motion-1.0 repository and pull LFS files
WORKDIR /app
RUN git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git /app/hy-motion && \
    cd /app/hy-motion && git lfs pull

# Install HY-Motion requirements (PyTorch already installed in base image)
WORKDIR /app/hy-motion
RUN pip install --no-cache-dir \
    --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple \
    -r requirements.txt

# Install RunPod SDK
RUN pip install --no-cache-dir runpod>=1.6.0

# Copy our custom handler
COPY handler.py /app/handler.py

# Set working directory back to /app
WORKDIR /app

# Set the entrypoint for RunPod serverless
CMD ["python", "-u", "handler.py"]
