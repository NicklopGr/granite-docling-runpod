# Granite-Docling-258M RunPod Serverless

Production-quality document understanding using IBM's Granite-Docling-258M with the official Docling SDK VlmPipeline.

## Features

- **97% Table TEDS Accuracy**: Best-in-class table structure recognition
- **IBM Production Approach**: Uses official Docling SDK with VlmPipeline
- **GPU Accelerated**: CUDA support for fast inference
- **RunPod Serverless**: Auto-scaling with webhook deployment

## Model Specs

| Metric | Value |
|--------|-------|
| Parameters | 258M |
| Table TEDS (structure) | 97% |
| Table TEDS (content) | 96% |
| Full-page OCR F1 | 0.84 |
| VRAM Usage | ~4GB |

## API

### Input
```json
{
  "input": {
    "pdf_base64": "base64_encoded_pdf_content"
  }
}
```

### Output
```json
{
  "status": "success",
  "result": {
    "markdown": "# Document Title\n\n| Col1 | Col2 |\n|---|---|\n| A | B |",
    "tables": [
      {
        "table_number": 1,
        "markdown": "| Col1 | Col2 |\n|---|---|\n| A | B |",
        "row_count": 2
      }
    ],
    "text_content": [
      {"text": "Document Title", "type": "title"},
      {"text": "Paragraph text...", "type": "paragraph"}
    ],
    "metadata": {
      "model": "granite-docling-258M",
      "pipeline": "VlmPipeline",
      "inference_time_seconds": 5.2,
      "total_time_seconds": 6.1,
      "table_count": 3,
      "text_items": 25
    }
  }
}
```

## Architecture

```
PDF → Docling SDK → VlmPipeline → Granite-Docling-258M → DocTags → Markdown/Tables
```

This uses the IBM-recommended production approach:
1. **Docling SDK**: Official document processing library
2. **VlmPipeline**: Vision-Language Model pipeline for 97% accuracy
3. **Granite-Docling-258M**: Compact 258M parameter VLM optimized for documents

## Deployment

This repo is connected to RunPod with webhook auto-build. Push to main triggers automatic deployment.

```bash
git add -A && git commit -m "Update message" && git push
```

## Local Testing

```bash
# Run container locally
docker run --gpus all -p 8000:8000 granite-docling-runpod

# Test with curl (PDF must be base64 encoded)
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"input": {"pdf_base64": "YOUR_BASE64_PDF"}}'
```

## References

- [Granite-Docling-258M on HuggingFace](https://huggingface.co/ibm-granite/granite-docling-258M)
- [Docling VLM Pipeline Example](https://docling-project.github.io/docling/examples/minimal_vlm_pipeline/)
- [IBM Announcement](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion)
