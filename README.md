# Granite-Docling-258M RunPod Serverless

Production-grade Granite-Docling-258M serving that mirrors IBM’s recommended direct **vLLM** deployment:

- PDFs are rendered at 240 DPI, streamed to `/tmp`, and fed page-by-page into `vllm.LLM`.
- `docling-core` ingests the complete set of DocTags + page images to rebuild continuation tables with cross-page heuristics.
- No Docling SDK server is embedded here—the FIN-OCR backend simply hits this RunPod endpoint.

## Features

- **97% Table TEDS Accuracy**: Granite-Docling VLM tuned for financial docs
- **Direct vLLM Inference**: Mirrors Docling’s inline VLM implementation (untied weights, transformers backend, eager mode)
- **docling-core Rendering**: Converts DocTags to Markdown, page slices, and structured tables on-GPU
- **RunPod Serverless**: Auto-scaling GPU endpoint with webhook-driven deploys

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
    "page_markdowns": [
      "# Document Title\n\n| Col1 | Col2 |\n|---|---|\n| A | B |"
    ],
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
      "pipeline": "direct-vllm",
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
PDF (base64) → handler.py
  ↳ pdf2image (240 DPI RGB written to /tmp)
  ↳ vLLM single-page batches (ibm-granite/granite-docling-258M, transformers backend, enforce_eager)
  ↳ docling-core multi-page parse (DocTagsDocument + DoclingDocument)
  ↳ Markdown + per-page markdown + tables + text content
```

Key runtime settings (see `handler.py`):
1. `model_impl="transformers"` and `revision="untied"` – required by IBM for Granite-Docling.
2. `enforce_eager=True` – avoids CUDA graph warmups (>10 min) on serverless cold starts.
3. `limit_mm_per_prompt={"image": 1}` and `enable_prefix_caching=False` – ensure multimodal stability.
4. Streaming 240 DPI PNGs to `/tmp` prevents 20+ GB spikes when processing 30+ page statements.

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
