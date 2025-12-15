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

# Wait for vLLM to be ready with comprehensive health check
echo "[Services] Waiting for vLLM to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "[Services] vLLM /v1/models endpoint is up!"

        # Test with actual completion request
        echo "[Services] Testing vLLM with sample request..."
        TEST_RESPONSE=$(curl -s -X POST http://localhost:8001/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "ibm-granite/granite-docling-258M",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }')

        if echo "$TEST_RESPONSE" | grep -q "choices"; then
            echo "[Services] vLLM server is fully operational!"
            echo "[Services] Test response: $TEST_RESPONSE"
            break
        else
            echo "[Services] vLLM models endpoint up but completion test failed"
            echo "[Services] Response: $TEST_RESPONSE"
        fi
    fi
    echo "[Services] Waiting for vLLM... ($i/60)"
    sleep 5
done

# Final verification
echo "[Services] Final vLLM verification..."
curl -s http://localhost:8001/v1/models || echo "[Services] ERROR: vLLM not responding!"

# Start RunPod handler
echo "[Services] Starting RunPod handler..."
python /app/handler.py

# If handler exits, kill vLLM
echo "[Services] Handler exited, stopping vLLM..."
kill $VLLM_PID
