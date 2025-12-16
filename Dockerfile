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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and build tools, then install flash-attn in same layer
# This ensures CUDA_HOME environment is available during compilation
RUN pip install --upgrade pip setuptools wheel && \
    echo "=== Environment Check ===" && \
    echo "CUDA_HOME: $CUDA_HOME" && \
    echo "PATH: $PATH" && \
    nvcc --version && \
    echo "=== Installing flash-attn ===" && \
    pip install ninja packaging && \
    MAX_JOBS=4 pip install flash-attn --no-build-isolation -v && \
    echo "=== Verifying flash-attn installation ===" && \
    python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')" || echo "flash-attn import failed"

# Install vLLM for direct inference (IBM's recommended production approach)
RUN pip install vllm

# Install dependencies for PDF rendering and processing
RUN pip install \
    pdf2image \
    pillow \
    transformers \
    runpod \
    accelerate

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

# Expose ports (8000 for RunPod, 8001 for vLLM)
EXPOSE 8000 8001

# Run startup script (starts vLLM + handler)
CMD ["/app/start-services.sh"]
