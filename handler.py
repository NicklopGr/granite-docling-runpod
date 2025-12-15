"""
Granite-Docling-258M RunPod Serverless Handler

Uses IBM Docling SDK with VlmPipeline for production-quality document understanding.
This is the recommended approach per IBM documentation.

Key Configuration:
- vlm_model_specs.GRANITEDOCLING_TRANSFORMERS for explicit model selection
- CUDA acceleration with flash_attention_2
- generate_page_images=True for better table extraction

API Input: {"input": {"pdf_base64": "base64_encoded_pdf"}}
API Output: {"status": "success", "result": {"markdown": "...", "tables": [...], "text_content": [...]}}

Reference:
- https://huggingface.co/ibm-granite/granite-docling-258M
- https://docling-project.github.io/docling/examples/minimal_vlm_pipeline/
- https://docling-project.github.io/docling/examples/gpu_vlm_pipeline/
"""

import runpod
import base64
import os
import time
import tempfile
import traceback

# Global converter - loaded once, reused across requests
converter = None


def check_flash_attn_available():
    """Check if flash_attn is installed and available."""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def load_converter():
    """
    Load Docling DocumentConverter with VlmPipeline for Granite-Docling.
    Uses explicit model specification and GPU acceleration.
    """
    global converter

    if converter is None:
        from docling.datamodel import vlm_model_specs
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        print("[GraniteDocling] Loading Docling with VlmPipeline...")
        print("[GraniteDocling] Using GRANITEDOCLING_TRANSFORMERS model spec")
        start_time = time.time()

        # Check if flash_attn is available
        use_flash_attn = check_flash_attn_available()
        print(f"[GraniteDocling] Flash Attention 2: {'enabled' if use_flash_attn else 'disabled (not installed)'}")

        # Configure VLM pipeline with explicit model and GPU acceleration
        pipeline_options = VlmPipelineOptions(
            # Explicitly use Granite-Docling with Transformers framework
            vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
            # Generate page images for better table extraction
            generate_page_images=True,
            # Configure GPU acceleration (only enable flash_attn if available)
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.CUDA,
                cuda_use_flash_attention2=use_flash_attn,
            ),
        )

        # Create converter with VlmPipeline
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )

        elapsed = time.time() - start_time
        print(f"[GraniteDocling] Converter loaded in {elapsed:.2f}s")

    return converter


def extract_tables_from_markdown(markdown: str) -> list:
    """Extract table sections from markdown content."""
    if not markdown:
        return []

    tables = []
    lines = markdown.split('\n')
    current_table = []
    table_num = 0

    for line in lines:
        if line.strip().startswith('|'):
            current_table.append(line)
        elif current_table:
            # End of table
            table_num += 1
            tables.append({
                "table_number": table_num,
                "markdown": '\n'.join(current_table),
                "row_count": len([l for l in current_table if l.strip().startswith('|') and '---' not in l])
            })
            current_table = []

    # Handle last table if exists
    if current_table:
        table_num += 1
        tables.append({
            "table_number": table_num,
            "markdown": '\n'.join(current_table),
            "row_count": len([l for l in current_table if l.strip().startswith('|') and '---' not in l])
        })

    return tables


def handler(event):
    """
    RunPod serverless handler for Granite-Docling inference using Docling SDK.
    """
    start_time = time.time()

    try:
        input_data = event.get("input", {})
        pdf_base64 = input_data.get("pdf_base64")

        if not pdf_base64:
            return {"status": "error", "error": "No pdf_base64 provided"}

        # Decode PDF
        print("[GraniteDocling] Decoding PDF...")
        pdf_bytes = base64.b64decode(pdf_base64)
        print(f"[GraniteDocling] PDF size: {len(pdf_bytes)} bytes")

        # Save to temp file (Docling requires file path)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name

        try:
            # Load converter
            doc_converter = load_converter()

            # Convert document with explicit parameters
            print("[GraniteDocling] Converting document with VlmPipeline...")
            print("[GraniteDocling] Config: max_num_pages=None (process all pages)")
            inference_start = time.time()

            # Convert document (no max_num_pages parameter = process all pages)
            result = doc_converter.convert(source=tmp_path)
            doc = result.document

            inference_time = time.time() - inference_start

            # Log output page count
            page_count = len(doc.pages) if hasattr(doc, 'pages') else 0
            print(f"[GraniteDocling] Conversion completed in {inference_time:.2f}s")
            print(f"[GraniteDocling] Processed {page_count} pages")

            # Export to markdown
            markdown = doc.export_to_markdown()
            print(f"[GraniteDocling] Markdown length: {len(markdown)} chars")

            # Extract tables from document structure (more accurate than parsing markdown)
            tables = []
            if hasattr(doc, 'tables') and doc.tables:
                for i, table in enumerate(doc.tables):
                    table_md = table.export_to_markdown() if hasattr(table, 'export_to_markdown') else str(table)
                    tables.append({
                        "table_number": i + 1,
                        "markdown": table_md,
                        "row_count": table_md.count('\n') if table_md else 0
                    })
            else:
                # Fallback to extracting from markdown
                tables = extract_tables_from_markdown(markdown)

            print(f"[GraniteDocling] Found {len(tables)} tables")

            # Extract text content with proper labels
            text_content = []
            if hasattr(doc, 'texts'):
                for item in doc.texts:
                    text_content.append({
                        "text": item.text if hasattr(item, 'text') else str(item),
                        "type": str(item.label) if hasattr(item, 'label') else "text"
                    })

            total_time = time.time() - start_time

            return {
                "status": "success",
                "result": {
                    "markdown": markdown,
                    "tables": tables,
                    "text_content": text_content,
                    "metadata": {
                        "model": "granite-docling-258M",
                        "pipeline": "VlmPipeline",
                        "framework": "transformers",
                        "accelerator": "cuda",
                        "inference_time_seconds": round(inference_time, 2),
                        "total_time_seconds": round(total_time, 2),
                        "table_count": len(tables),
                        "text_items": len(text_content),
                        "page_count": page_count
                    }
                }
            }

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        print(f"[GraniteDocling] Error: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    print("[GraniteDocling] Starting RunPod serverless handler with Docling SDK...")
    print("[GraniteDocling] Model: GRANITEDOCLING_TRANSFORMERS")
    print("[GraniteDocling] Accelerator: CUDA with flash_attention_2")
    runpod.serverless.start({"handler": handler})
