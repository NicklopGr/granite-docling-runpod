# Granite-Docling-258M RunPod Serverless

RunPod serverless endpoint for IBM's [Granite-Docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) vision-language model.

## Features

- **VLLM inference**: ~500 tokens/sec (100x faster than transformers)
- **DocTags to HTML**: Automatic conversion for easy downstream processing
- **Table extraction**: Extracts HTML tables from documents
- **Pre-loaded model**: Reduces cold start time

## Model Specs

| Metric | Value |
|--------|-------|
| Parameters | 258M |
| Table TEDS (structure) | 97% |
| Table TEDS (content) | 96% |
| Full-page OCR F1 | 0.84 |
| VRAM Usage | ~2GB |

## API

### Input
```json
{
  "input": {
    "image_base64": "base64_encoded_image_string"
  }
}
```

### Output
```json
{
  "status": "success",
  "result": {
    "doctags": "raw DocTags output",
    "html": "converted HTML",
    "tables": [
      {"table_number": 1, "html": "<table>...</table>"}
    ],
    "metadata": {
      "model": "granite-docling-258M",
      "inference_time_seconds": 2.5,
      "total_time_seconds": 3.2,
      "table_count": 2
    }
  }
}
```

## Deployment

### Build Docker Image
```bash
docker build -t granite-docling-runpod .
```

### Push to Docker Hub
```bash
docker tag granite-docling-runpod your-dockerhub/granite-docling-runpod:latest
docker push your-dockerhub/granite-docling-runpod:latest
```

### Create RunPod Endpoint
1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Create new endpoint
3. Use Docker image: `your-dockerhub/granite-docling-runpod:latest`
4. GPU: RTX 4090 or A100 (24GB+ VRAM recommended)
5. Save endpoint ID for backend integration

## Local Testing

```bash
# Run container locally
docker run --gpus all -p 8000:8000 granite-docling-runpod

# Test with curl
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"input": {"image_base64": "YOUR_BASE64_IMAGE"}}'
```

## Notes

- Uses `revision="untied"` for VLLM compatibility (tied weights not fully supported)
- Model is pre-downloaded during Docker build to reduce cold starts
- Outputs DocTags format natively, converted to HTML by docling-core library
