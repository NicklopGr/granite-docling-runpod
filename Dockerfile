# Granite-Docling-258M RunPod Serverless Container
# Uses VLLM for fast inference (~500 tokens/sec)
#
# Build: docker build -t granite-docling-runpod .
# Test locally: docker run --gpus all -p 8000:8000 granite-docling-runpod

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install VLLM with CUDA support
RUN pip install vllm>=0.6.0

# Install docling-core for DocTags conversion
RUN pip install docling-core

# Install other dependencies
RUN pip install \
    pillow \
    runpod \
    transformers \
    accelerate

# Pre-download the model during build (reduces cold start time)
# Using the "untied" revision for VLLM compatibility
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ibm-granite/granite-docling-258M', revision='untied')"

# Copy handler
WORKDIR /app
COPY handler.py /app/handler.py

# Health check endpoint (optional)
EXPOSE 8000

# Run the handler
CMD ["python", "/app/handler.py"]
