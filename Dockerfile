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
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and build tools
RUN pip install --upgrade pip setuptools wheel

# Cache-busting argument to force rebuild of flash-attn layer
ARG CACHEBUST=1
RUN echo "Cache bust: $CACHEBUST - Timestamp: $(date -u +%Y%m%d-%H%M%S)"

# Install flash-attn for CUDA acceleration (requires ninja for build)
# Use MAX_JOBS to speed up compilation
RUN pip install ninja packaging && \
    MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Install Docling with VLM support (IBM production approach)
# Use latest version 2.64.1 (Dec 2025) with all bug fixes
RUN pip install "docling[vlm]==2.64.1"

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
