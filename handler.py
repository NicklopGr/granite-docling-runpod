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

Build: 2025-12-15-v16 (Proper dict iteration for doc.pages)
"""

import runpod
import base64
import os
import time
import tempfile
import traceback
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global converter - loaded once, reused across requests
converter = None


def check_flash_attn_available():
    """Check if flash_attn is installed and available."""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def check_vllm_health():
    """Verify vLLM server is ready and can process requests."""
    import requests
    try:
        # Check models endpoint
        response = requests.get("http://localhost:8001/v1/models", timeout=5)
        if response.status_code != 200:
            logger.error(f"vLLM models endpoint returned {response.status_code}")
            return False

        models = response.json()
        logger.info(f"vLLM models available: {models}")

        # Test with simple completion request
        test_response = requests.post(
            "http://localhost:8001/v1/chat/completions",
            json={
                "model": "ibm-granite/granite-docling-258M",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            },
            timeout=30
        )

        if test_response.status_code != 200:
            logger.error(f"vLLM test request failed: {test_response.status_code} - {test_response.text}")
            return False

        logger.info("vLLM health check PASSED")
        return True

    except Exception as e:
        logger.error(f"vLLM health check failed: {e}")
        return False


def setup_api_logging():
    """Monkey-patch requests to log all vLLM API calls and convert RGBA to RGB."""
    import requests
    import re
    from PIL import Image
    import io
    original_post = requests.post

    def logged_post(url, *args, **kwargs):
        if "localhost:8001" in url:
            logger.info(f"=== vLLM API Request ===")
            logger.info(f"URL: {url}")

            # CRITICAL: Log max_tokens to verify it's being sent!
            if 'json' in kwargs:
                logger.info(f"[Request Params] model: {kwargs['json'].get('model', 'NOT SET')}")
                logger.info(f"[Request Params] max_tokens: {kwargs['json'].get('max_tokens', 'NOT SET - WILL DEFAULT TO 16!')}")
                logger.info(f"[Request Params] temperature: {kwargs['json'].get('temperature', 'NOT SET')}")
                logger.info(f"[Request Params] skip_special_tokens: {kwargs['json'].get('skip_special_tokens', 'NOT SET')}")

            # Log request body (truncate images) and fix RGBA issue
            if 'json' in kwargs:
                req_data = kwargs['json'].copy()
                if 'messages' in req_data:
                    for msg in req_data['messages']:
                        if 'content' in msg and isinstance(msg['content'], list):
                            for item in msg['content']:
                                if item.get('type') == 'image_url':
                                    url_str = item['image_url']['url']
                                    if url_str.startswith('data:'):
                                        # Validate base64 encoding
                                        parts = url_str.split(',', 1)
                                        if len(parts) == 2:
                                            header, b64_data = parts
                                            logger.info(f"[Base64 Validation] Data URI header: {header}")
                                            logger.info(f"[Base64 Validation] Base64 length: {len(b64_data)} chars")
                                            logger.info(f"[Base64 Validation] First 100 chars: {b64_data[:100]}")
                                            logger.info(f"[Base64 Validation] Last 50 chars: {b64_data[-50:]}")

                                            # Check for invalid characters
                                            invalid_chars = re.findall(r'[^A-Za-z0-9+/=]', b64_data)
                                            if invalid_chars:
                                                logger.error(f"[Base64 Validation] INVALID CHARACTERS FOUND: {set(invalid_chars)}")
                                                logger.error(f"[Base64 Validation] Character positions: {[(c, b64_data.index(c)) for c in set(invalid_chars)]}")
                                            else:
                                                logger.info(f"[Base64 Validation] Base64 string appears valid (no invalid chars)")

                                            # Check for common issues
                                            if '\n' in b64_data or '\r' in b64_data:
                                                logger.error(f"[Base64 Validation] Contains newlines!")
                                            if ' ' in b64_data:
                                                logger.error(f"[Base64 Validation] Contains spaces!")

                                            # FIX: Convert RGBA to RGB for vLLM compatibility
                                            try:
                                                img_bytes = base64.b64decode(b64_data)
                                                img = Image.open(io.BytesIO(img_bytes))
                                                logger.info(f"[Image Fix] Original format: {img.mode}, size: {img.size}")

                                                if img.mode == 'RGBA':
                                                    logger.warning(f"[Image Fix] Converting RGBA to RGB (vLLM doesn't support RGBA)")
                                                    # Create white background
                                                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                                                    # Paste RGBA image onto white background
                                                    rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask

                                                    # Re-encode as base64
                                                    buffer = io.BytesIO()
                                                    rgb_img.save(buffer, format='PNG')
                                                    new_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                                                    # Update the request
                                                    item['image_url']['url'] = f"data:image/png;base64,{new_b64}"
                                                    logger.info(f"[Image Fix] Converted to RGB, new base64 length: {len(new_b64)}")

                                                    # Update for logging
                                                    url_str = item['image_url']['url']
                                                    b64_data = new_b64
                                            except Exception as e:
                                                logger.error(f"[Image Fix] Failed to convert image: {e}")

                                        else:
                                            logger.error(f"[Base64 Validation] Data URI format invalid: {url_str[:100]}")

        response = original_post(url, *args, **kwargs)

        if "localhost:8001" in url:
            logger.info(f"=== vLLM API Response ===")
            logger.info(f"Status: {response.status_code}")

            # Log response
            try:
                resp_json = response.json()
                logger.info(f"Response: {json.dumps(resp_json, indent=2)[:2000]}")  # First 2000 chars
            except:
                logger.info(f"Response (raw): {response.text[:500]}")

            # Log request summary (after response, so we don't corrupt the request)
            if 'json' in kwargs and 'messages' in kwargs['json']:
                for msg in kwargs['json']['messages']:
                    if isinstance(msg.get('content'), list):
                        for item in msg['content']:
                            if item.get('type') == 'image_url' and 'image_url' in item:
                                url_str = item['image_url'].get('url', '')
                                if url_str.startswith('data:'):
                                    parts = url_str.split(',', 1)
                                    if len(parts) == 2:
                                        logger.info(f"Request included image: data:image/png;base64,...({len(parts[1])} chars)")

        return response

    requests.post = logged_post
    logger.info("API logging enabled")


def load_converter():
    """
    Load Docling DocumentConverter with VlmPipeline for Granite-Docling.
    Uses external vLLM server via API for optimal performance.
    """
    global converter

    if converter is None:
        from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions, TableStructureOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline

        logger.info("[GraniteDocling] Loading Docling with VlmPipeline...")
        logger.info("[GraniteDocling] Using external vLLM server (http://localhost:8001)")

        # Health check BEFORE loading converter
        if not check_vllm_health():
            raise RuntimeError("vLLM server health check failed - cannot initialize converter")

        # Enable API logging
        setup_api_logging()

        start_time = time.time()

        # Configure VLM pipeline with external vLLM API
        vlm_options = ApiVlmOptions(
            url="http://localhost:8001/v1/chat/completions",
            params=dict(
                model="ibm-granite/granite-docling-258M",
                max_tokens=6144,  # Increased from 4096 - model context is 8192, images use ~1000-2000 tokens
                skip_special_tokens=False,  # CRITICAL for DOCTAGS parsing
            ),
            headers={},
            prompt="Convert this page to docling.",
            timeout=600,  # Increased to 10 minutes
            scale=2.0,
            temperature=0.0,
            response_format=ResponseFormat.DOCTAGS,
        )

        # Configure table structure options - disable cell matching for repetitive text
        table_structure_options = TableStructureOptions(
            do_cell_matching=False  # CRITICAL: Prevents cell boundary confusion with repetitive text (e.g., "TURO8667352901" duplicates)
        )

        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            # Table structure configuration
            table_structure_options=table_structure_options,
            # CRITICAL: Use backend text extraction instead of VLM-generated text (prevents hallucination)
            force_backend_text=True,
            # Generate page images for better table extraction
            generate_page_images=True,
            # Image resolution scale (higher = better quality, but slower)
            images_scale=2.0,
            # Enable remote services for external vLLM API
            enable_remote_services=True,
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
        logger.info(f"[GraniteDocling] Converter loaded in {elapsed:.2f}s")

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

            # VALIDATION: Check if page images were generated
            # Note: doc.pages is a dict (not list) mapping page_no -> PageItem
            if hasattr(doc, 'pages') and doc.pages:
                page_count = len(doc.pages)
                logger.info(f"[GraniteDocling] Generated {page_count} page objects")
                # Get first 3 page objects from the dict (doc.pages.items() returns (page_no, PageItem) tuples)
                for i, (page_no, page) in enumerate(list(doc.pages.items())[:3]):
                    logger.info(f"[GraniteDocling] Page {page_no}: Checking image...")
                    if hasattr(page, 'image'):
                        img = page.image
                        if img is not None:
                            logger.info(f"[GraniteDocling] Page {page_no}: Image size {img.size if hasattr(img, 'size') else 'unknown'}")
                        else:
                            logger.warning(f"[GraniteDocling] Page {page_no}: Image is None!")
                    else:
                        logger.warning(f"[GraniteDocling] Page {page_no}: No image attribute")
            else:
                page_count = 0
                logger.error("[GraniteDocling] No pages in document!")

            logger.info(f"[GraniteDocling] Conversion completed in {inference_time:.2f}s")
            logger.info(f"[GraniteDocling] Processed {page_count} pages")

            # Export to markdown
            markdown = doc.export_to_markdown()
            logger.info(f"[GraniteDocling] Markdown length: {len(markdown)} chars")

            # Extract tables from document structure (more accurate than parsing markdown)
            logger.info("[GraniteDocling] Inspecting document structure...")
            logger.info(f"[GraniteDocling] Document attributes: {dir(doc)}")

            tables = []
            if hasattr(doc, 'tables') and doc.tables:
                logger.info(f"[GraniteDocling] Found {len(doc.tables)} table objects")
                for i, table in enumerate(doc.tables):
                    logger.info(f"[GraniteDocling] Table {i}: {type(table)} - {dir(table)}")
                    table_md = table.export_to_markdown() if hasattr(table, 'export_to_markdown') else str(table)
                    tables.append({
                        "table_number": i + 1,
                        "markdown": table_md,
                        "row_count": table_md.count('\n') if table_md else 0
                    })
            else:
                logger.warning("[GraniteDocling] No tables found in document structure")
                # Fallback to extracting from markdown
                tables = extract_tables_from_markdown(markdown)

            # Validate markdown content
            if not markdown or len(markdown.strip()) == 0:
                logger.error("[GraniteDocling] CRITICAL: Markdown is empty!")
                logger.error(f"[GraniteDocling] Document type: {type(doc)}")
                logger.error(f"[GraniteDocling] Document dir: {dir(doc)}")
                if hasattr(doc, 'texts'):
                    logger.error(f"[GraniteDocling] doc.texts length: {len(doc.texts) if doc.texts else 0}")
            else:
                logger.info(f"[GraniteDocling] Markdown length: {len(markdown)} chars (first 500): {markdown[:500]}")

            logger.info(f"[GraniteDocling] Found {len(tables)} tables")

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
    print("[GraniteDocling] Model: External vLLM server (localhost:8001)")
    print("[GraniteDocling] Revision: untied, dtype: float32")
    runpod.serverless.start({"handler": handler})
