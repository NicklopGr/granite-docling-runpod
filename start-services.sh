#!/bin/bash
# Start RunPod handler with direct transformers inference
# No separate inference server needed - model loaded directly in handler.py

echo "[Services] Starting RunPod handler with transformers inference..."
echo "[Services] Model: ibm-granite/granite-docling-258M (untied)"
echo "[Services] Approach: Direct transformers client (Docling inline pipeline)"

# Start RunPod handler
python /app/handler.py
