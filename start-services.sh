#!/bin/bash
# Start RunPod handler with direct vLLM inference
# No separate vLLM server needed - model loaded directly in handler.py

echo "[Services] Starting RunPod handler with direct vLLM inference..."
echo "[Services] Model: ibm-granite/granite-docling-258M (untied)"
echo "[Services] Approach: Direct vLLM.LLM() client (IBM production recommendation)"

# Start RunPod handler
python /app/handler.py
