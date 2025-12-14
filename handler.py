"""
Granite-Docling-258M RunPod Serverless Handler

Uses IBM Docling SDK with VlmPipeline for production-quality document understanding.
This is the recommended approach per IBM documentation.

API Input: {"input": {"pdf_base64": "base64_encoded_pdf"}}
API Output: {"status": "success", "result": {"markdown": "...", "tables": [...], "text_content": [...]}}

Reference:
- https://huggingface.co/ibm-granite/granite-docling-258M
- https://docling-project.github.io/docling/examples/minimal_vlm_pipeline/
"""

import runpod
import base64
import io
import os
import re
import time
import tempfile
import traceback

# Global converter - loaded once, reused across requests
converter = None


def load_converter():
    """
    Load Docling DocumentConverter with VlmPipeline for Granite-Docling.
    This uses the IBM production-recommended approach.
    """
    global converter

    if converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        print("[GraniteDocling] Loading Docling with VlmPipeline...")
        start_time = time.time()

        # Create converter with VlmPipeline (uses Granite-Docling by default)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
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
    # Find markdown tables (lines starting with |)
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

            # Convert document
            print("[GraniteDocling] Converting document with VlmPipeline...")
            inference_start = time.time()

            result = doc_converter.convert(source=tmp_path)
            doc = result.document

            inference_time = time.time() - inference_start
            print(f"[GraniteDocling] Conversion completed in {inference_time:.2f}s")

            # Export to markdown
            markdown = doc.export_to_markdown()
            print(f"[GraniteDocling] Markdown length: {len(markdown)} chars")

            # Extract tables
            tables = extract_tables_from_markdown(markdown)
            print(f"[GraniteDocling] Found {len(tables)} tables")

            # Extract text content
            text_content = []
            for item in doc.texts:
                text_content.append({
                    "text": item.text if hasattr(item, 'text') else str(item),
                    "type": item.label if hasattr(item, 'label') else "text"
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
                        "inference_time_seconds": round(inference_time, 2),
                        "total_time_seconds": round(total_time, 2),
                        "table_count": len(tables),
                        "text_items": len(text_content)
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
    runpod.serverless.start({"handler": handler})
