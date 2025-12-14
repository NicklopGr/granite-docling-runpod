# Granite-Docling-258M RunPod Serverless Container
# Uses Transformers + Flash Attention 2 for reliable inference
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

# Install Flash Attention 2 for faster inference
RUN pip install flash-attn --no-build-isolation

# Install core dependencies
RUN pip install \
    transformers>=4.40.0 \
    accelerate \
    pillow \
    runpod \
    docling-core \
    torch

# Pre-download the model during build (reduces cold start time)
RUN python -c "\
from transformers import AutoProcessor, AutoModelForVision2Seq; \
AutoProcessor.from_pretrained('ibm-granite/granite-docling-258M'); \
AutoModelForVision2Seq.from_pretrained('ibm-granite/granite-docling-258M')"

# Copy handler
WORKDIR /app
COPY handler.py /app/handler.py

# Health check endpoint (optional)
EXPOSE 8000

# Run the handler
CMD ["python", "/app/handler.py"]
