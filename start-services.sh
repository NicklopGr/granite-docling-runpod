#!/bin/bash
# Start vLLM server in background, then start RunPod handler

echo "[Services] Starting vLLM server on port 8001..."
vllm serve ibm-granite/granite-docling-258M \
    --revision untied \
    --dtype float32 \
    --port 8001 \
    --host 0.0.0.0 &

VLLM_PID=$!
echo "[Services] vLLM server started with PID $VLLM_PID"

# Wait for vLLM to be ready (check health endpoint)
echo "[Services] Waiting for vLLM to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "[Services] vLLM server is ready!"
        break
    fi
    echo "[Services] Waiting for vLLM... ($i/60)"
    sleep 5
done

# Start RunPod handler
echo "[Services] Starting RunPod handler..."
python /app/handler.py

# If handler exits, kill vLLM
echo "[Services] Handler exited, stopping vLLM..."
kill $VLLM_PID
