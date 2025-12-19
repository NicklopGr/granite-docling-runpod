# Granite-Docling-258M RunPod Serverless Container
# Uses HuggingFace transformers (Docling inline pipeline) for production-quality document understanding
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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and build tools
RUN pip install --upgrade pip setuptools wheel

# Install dependencies for PDF rendering and processing
# v38: Pin docling-core==2.55.0 (latest public build) for improved continuation tables
RUN pip install \
    pdf2image \
    pillow \
    transformers \
    runpod \
    accelerate \
    docling-core==2.55.0

# Pre-download the Granite-Docling model during build (reduces cold start time)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ibm-granite/granite-docling-258M'); \
print('Granite-Docling-258M downloaded successfully')"

# Copy handler and startup script
WORKDIR /app
COPY handler.py /app/handler.py
COPY start-services.sh /app/start-services.sh
RUN chmod +x /app/start-services.sh

# Expose port for RunPod
EXPOSE 8000

# Run startup script (starts transformers handler)
CMD ["/app/start-services.sh"]
