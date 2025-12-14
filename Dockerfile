# Granite-Docling-258M RunPod Serverless Container
# Uses IBM Docling SDK with VlmPipeline for production-quality document understanding
#
# Build: docker build -t granite-docling-runpod .
# Test locally: docker run --gpus all -p 8000:8000 granite-docling-runpod

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV DOCLING_DEVICE=cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Docling with VLM support (IBM production approach)
RUN pip install "docling[vlm]"

# Install additional dependencies
RUN pip install \
    pillow \
    runpod \
    accelerate

# Pre-download the Granite-Docling model during build (reduces cold start time)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ibm-granite/granite-docling-258M'); \
print('Granite-Docling-258M downloaded successfully')"

# Copy handler
WORKDIR /app
COPY handler.py /app/handler.py

# Health check endpoint (optional)
EXPOSE 8000

# Run the handler
CMD ["python", "/app/handler.py"]
