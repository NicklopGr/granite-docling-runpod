"""
Granite-Docling-258M RunPod Serverless Handler (Direct vLLM Inference)

Uses direct vLLM API following IBM's recommended production approach.
Uses docling-core for proper DocTags -> Markdown conversion.

Key Features:
- Direct vLLM.LLM() client with untied weights
- PDF rendered to RGB using pdf2image (no RGBA conversion)
- docling-core for DocTags parsing (DocTagsDocument + DoclingDocument)
- 144 DPI rendering (scale=2.0 × 72 base DPI, matches Docling defaults)
- Max 10 pages per vLLM batch (memory optimization)
- v37: Multi-page docling-core parsing (all pages at once for cross-page table context)

API Input: {"input": {"pdf_base64": "base64_encoded_pdf"}}
API Output: {"status": "success", "result": {"markdown": "...", "tables": [...], "text_content": [...]}}

Reference:
- https://huggingface.co/ibm-granite/granite-docling-258M
- IBM recommendation: Direct vLLM for production (not Docling SDK)

Build: 2025-12-17-v38-debug-tables
"""

import runpod
import base64
import os
import time
import traceback
import json
import logging
import re
from typing import List, Dict, Any
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# v35: Enable TensorFloat32 tensor cores for faster float32 matrix multiplication
# This addresses the warning: "TensorFloat32 tensor cores available but not enabled"
# TF32 uses 19-bit precision (vs 32-bit) for 8x faster matmul with minimal accuracy loss
torch.set_float32_matmul_precision('high')
logger.info("[GraniteDocling] TF32 tensor cores enabled (float32_matmul_precision='high')")

# Global vLLM client - loaded once, reused across requests
llm = None
processor = None


def load_vllm():
    """Load vLLM with Granite-Docling-258M (untied weights)."""
    global llm, processor

    if llm is None:
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        logger.info("[GraniteDocling] Loading vLLM with direct inference...")
        logger.info("[GraniteDocling] Model: ibm-granite/granite-docling-258M (untied)")

        start_time = time.time()

        # Load vLLM model with Docling-aligned settings
        # v34: Use vLLM defaults for dtype, gpu_memory_utilization, max_model_len
        #      Only set parameters that Docling explicitly requires:
        #      - model_impl="transformers": Forces HuggingFace backend for Idefics3
        #      - limit_mm_per_prompt: Required for multimodal models
        #      - enable_prefix_caching=False: Prevents KV cache reuse across different images
        #      See: https://github.com/docling-project/docling/blob/main/docling/models/vlm_models_inline/vllm_model.py
        # v36: Add enforce_eager=True to disable CUDA graph compilation
        # Docling's vllm_model.py uses this exact configuration - without it,
        # Transformers backend triggers 10+ minute CUDA graph compilation
        # See: https://github.com/docling-project/docling/blob/main/docling/models/vlm_models_inline/vllm_model.py
        llm = LLM(
            model="ibm-granite/granite-docling-258M",
            revision="untied",  # CRITICAL - untied weights required
            model_impl="transformers",  # CRITICAL - use Transformers backend for Idefics3
            enforce_eager=True,  # CRITICAL - disable CUDA graphs (matches Docling)
            limit_mm_per_prompt={"image": 1},  # CRITICAL - required for multimodal models
            trust_remote_code=True,
            enable_prefix_caching=False  # CRITICAL - disable for multimodal
            # Let vLLM auto-detect: dtype, gpu_memory_utilization (0.9), max_model_len
        )

        # Load processor for prompt formatting
        processor = AutoProcessor.from_pretrained(
            "ibm-granite/granite-docling-258M",
            revision="untied",
            trust_remote_code=True
        )

        elapsed = time.time() - start_time
        logger.info(f"[GraniteDocling] vLLM loaded in {elapsed:.2f}s")

    return llm, processor


def render_pdf_to_rgb(pdf_bytes: bytes) -> List[Image.Image]:
    """
    Render PDF pages directly to RGB (no RGBA conversion needed).

    Uses pdf2image with PPM format which produces RGB by default.
    """
    from pdf2image import convert_from_bytes

    logger.info("[GraniteDocling] Rendering PDF to RGB images...")
    start_time = time.time()

    # Convert PDF to images
    # v33: Changed from 150 DPI to 144 DPI to match Docling's scale=2.0 setting
    # Docling uses scale=2.0 which is 72 base DPI × 2 = 144 DPI
    # See: https://github.com/docling-project/docling/blob/main/docling/datamodel/vlm_model_specs.py
    images = convert_from_bytes(
        pdf_bytes,
        dpi=144,  # v33: Match Docling's scale=2.0 (72 × 2 = 144 DPI)
        fmt='ppm'  # PPM format = RGB by default (no alpha channel)
    )

    # Convert to RGB explicitly (in case any format creates RGBA)
    rgb_images = []
    for i, img in enumerate(images):
        if img.mode != 'RGB':
            logger.warning(f"[GraniteDocling] Page {i+1}: Converting {img.mode} to RGB")
            img = img.convert('RGB')
        else:
            logger.info(f"[GraniteDocling] Page {i+1}: Already RGB mode")

        logger.info(f"[GraniteDocling] Page {i+1}: Size {img.size}")
        rgb_images.append(img)

    elapsed = time.time() - start_time
    logger.info(f"[GraniteDocling] Rendered {len(rgb_images)} pages in {elapsed:.2f}s")

    return rgb_images


def parse_all_doctags_with_docling_core(all_doctags: List[str], all_images: List[Image.Image]) -> Dict[str, Any]:
    """
    Parse ALL DocTags outputs at once using docling-core library (IBM's official approach).

    v37: CRITICAL FIX - Parse entire document at once instead of page-by-page.
    Docling's VLM pipeline expects the full document so its global table heuristics
    can run across all pages. Page-by-page parsing was causing empty tables on pages 2+
    because continuation tables need context from page 1's header structure.

    This properly handles the location tokens (<loc_X>) and converts
    the raw DocTags output to structured Markdown with tables extracted.
    """
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument

    logger.info(f"[GraniteDocling] v37: Parsing {len(all_doctags)} pages with docling-core (multi-page mode)")

    try:
        # v37: Pass ALL pages at once to docling-core
        # This allows cross-page table heuristics to work properly
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(all_doctags, all_images)
        doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")

        # Export to markdown
        markdown = doc.export_to_markdown()

        # Extract tables from the document
        tables = []
        text_content = []

        # DoclingDocument has structured access to tables
        # v38: Added detailed debugging for table extraction failures
        logger.info(f"[GraniteDocling] v38: Found {len(doc.tables) if hasattr(doc, 'tables') and doc.tables else 0} tables in document")
        if hasattr(doc, 'tables') and doc.tables:
            for i, table in enumerate(doc.tables):
                try:
                    logger.info(f"[GraniteDocling] v38: Processing table {i+1}/{len(doc.tables)}")
                    logger.info(f"[GraniteDocling] v38: Table {i+1} type: {type(table)}")
                    logger.info(f"[GraniteDocling] v38: Table {i+1} has export_to_markdown: {hasattr(table, 'export_to_markdown')}")

                    # v38: Try different export methods
                    table_md = ""
                    if hasattr(table, 'export_to_markdown'):
                        try:
                            # Try with doc parameter first
                            table_md = table.export_to_markdown(doc=doc)
                            logger.info(f"[GraniteDocling] v38: Table {i+1} export_to_markdown(doc=doc) succeeded: {len(table_md)} chars")
                        except TypeError as te:
                            # Fallback without doc parameter if it's not supported
                            logger.warning(f"[GraniteDocling] v38: Table {i+1} export_to_markdown(doc=doc) failed: {te}")
                            try:
                                table_md = table.export_to_markdown()
                                logger.info(f"[GraniteDocling] v38: Table {i+1} export_to_markdown() succeeded: {len(table_md)} chars")
                            except Exception as e2:
                                logger.error(f"[GraniteDocling] v38: Table {i+1} export_to_markdown() also failed: {e2}")
                                table_md = str(table)
                        except Exception as e:
                            logger.error(f"[GraniteDocling] v38: Table {i+1} export_to_markdown(doc=doc) exception: {e}")
                            table_md = str(table)
                    else:
                        table_md = str(table)
                        logger.info(f"[GraniteDocling] v38: Table {i+1} used str(): {len(table_md)} chars")

                    row_count = len(table_md.strip().split('\n')) - 1 if table_md else 0
                    tables.append({
                        "table_number": i + 1,
                        "markdown": table_md,
                        "row_count": row_count
                    })
                    logger.info(f"[GraniteDocling] Table {i+1}: {row_count} rows")
                    if row_count == 0 or not table_md.strip():
                        logger.warning(f"[GraniteDocling] v38: Table {i+1} markdown empty after export (possible continuation page)")

                except Exception as table_error:
                    logger.exception(f"[GraniteDocling] v38: FAILED processing table {i+1}: {table_error}")
                    # Add empty table to maintain count
                    tables.append({
                        "table_number": i + 1,
                        "markdown": "",
                        "row_count": 0,
                        "error": str(table_error)
                    })

        # Extract text content
        if hasattr(doc, 'texts') and doc.texts:
            for text in doc.texts:
                text_content.append({
                    "text": str(text),
                    "type": "text"
                })

        logger.info(f"[GraniteDocling] docling-core parsed: {len(tables)} tables, {len(text_content)} text items")
        logger.info(f"[GraniteDocling] Markdown length: {len(markdown)} chars")

        return {
            "tables": tables,
            "text_content": text_content,
            "markdown": markdown,
            "raw_doctags": all_doctags
        }

    except Exception as e:
        logger.error(f"[GraniteDocling] docling-core parsing failed: {e}")
        logger.error(f"[GraniteDocling] Error details: {traceback.format_exc()}")
        logger.info("[GraniteDocling] Falling back to raw DocTags output")
        # Fallback: return raw doctags as text
        combined_doctags = "\n\n--- Page Break ---\n\n".join(all_doctags)
        return {
            "tables": [],
            "text_content": [{"text": combined_doctags, "type": "raw_doctags"}],
            "markdown": combined_doctags,
            "raw_doctags": all_doctags
        }


def combine_pages(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine results from multiple pages into single result."""
    all_tables = []
    all_text = []
    full_markdown = []

    table_counter = 0
    for page_num, page_result in enumerate(page_results):
        # Add page separator in markdown
        if page_num > 0:
            full_markdown.append(f"\n\n--- Page {page_num + 1} ---\n\n")

        # Add the page's markdown (from docling-core)
        page_markdown = page_result.get("markdown", "")
        if page_markdown:
            full_markdown.append(page_markdown)

        # Renumber tables globally
        for table in page_result.get("tables", []):
            table_counter += 1
            table["table_number"] = table_counter
            all_tables.append(table)

        # Add text content
        for text_item in page_result.get("text_content", []):
            all_text.append(text_item)

    return {
        "tables": all_tables,
        "text_content": all_text,
        "markdown": '\n\n'.join(full_markdown),
        "raw_doctags": [p.get("raw_doctags", "") for p in page_results]
    }


def process_pdf(pdf_base64: str) -> Dict[str, Any]:
    """
    Process PDF with direct vLLM inference using BATCH processing.

    Steps:
    1. Decode PDF bytes
    2. Render to RGB images (144 DPI, PPM format)
    3. Process ALL pages in batched llm.generate() calls
    4. v37: Collect ALL DocTags, then parse with docling-core ONCE (multi-page mode)
    5. Return combined result

    v31: Switched from sequential to batch processing.
    v37: Parse all pages at once with docling-core for cross-page table context.
    See: https://github.com/docling-project/docling/blob/main/docling/models/vlm_models_inline/vllm_model.py
    """
    from vllm import SamplingParams

    # Load vLLM client
    llm_client, proc = load_vllm()

    # Decode PDF
    logger.info("[GraniteDocling] Decoding PDF...")
    pdf_bytes = base64.b64decode(pdf_base64)
    logger.info(f"[GraniteDocling] PDF size: {len(pdf_bytes)} bytes")

    # Render to RGB
    rgb_images = render_pdf_to_rgb(pdf_bytes)

    # IBM's official message format for granite-docling
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."}
            ],
        },
    ]

    # Configure sampling
    # v33: Added stop strings from Docling's vlm_model_specs.py
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=8192,  # IBM's recommended value for full DOCTAGS output
        skip_special_tokens=False,  # Preserve DOCTAGS
        stop=["</doctag>", "<|end_of_text|>"]  # v33: Docling's stop strings
    )

    # v36: With enforce_eager (no CUDA graphs), memory is more predictable
    # Model is only 258M params (~600MB), can handle larger batches
    # RTX 4090 (24GB) / Ada 6000 (48GB) easily support 10+ page batches
    MAX_PAGES_PER_BATCH = 10

    logger.info(f"[GraniteDocling] Processing {len(rgb_images)} pages (max {MAX_PAGES_PER_BATCH} per batch)...")
    inference_start = time.time()

    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    logger.info(f"[GraniteDocling] Using prompt format: {prompt[:200]}...")

    # v37: Collect ALL doctags and images first, then parse together
    all_doctags = []
    all_images_for_parsing = []
    total_pages = len(rgb_images)

    # Process pages in chunks of MAX_PAGES_PER_BATCH (vLLM inference)
    for batch_start in range(0, total_pages, MAX_PAGES_PER_BATCH):
        batch_end = min(batch_start + MAX_PAGES_PER_BATCH, total_pages)
        batch_images = rgb_images[batch_start:batch_end]
        batch_num = (batch_start // MAX_PAGES_PER_BATCH) + 1
        total_batches = (total_pages + MAX_PAGES_PER_BATCH - 1) // MAX_PAGES_PER_BATCH

        logger.info(f"[GraniteDocling] Batch {batch_num}/{total_batches}: Pages {batch_start+1}-{batch_end}")

        # Build batch inputs for this chunk
        batched_inputs = []
        for i, image in enumerate(batch_images):
            batched_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            })
            logger.info(f"[GraniteDocling] Added page {batch_start + i + 1} to batch")

        # Single generate call for this batch
        logger.info(f"[GraniteDocling] Running batch inference for {len(batched_inputs)} pages...")
        outputs = llm_client.generate(batched_inputs, sampling_params=sampling_params)

        # v37: Collect outputs instead of parsing immediately
        for i, output in enumerate(outputs):
            page_num = batch_start + i + 1
            doctags = output.outputs[0].text

            logger.info(f"[GraniteDocling] Page {page_num}: {len(doctags)} chars")
            # DEBUG: Log first 500 chars of raw output to see actual format
            logger.info(f"[GraniteDocling] Page {page_num} raw output (first 500 chars): {doctags[:500]}")
            # DEBUG: Log last 200 chars to check if output is truncated
            logger.info(f"[GraniteDocling] Page {page_num} raw output (last 200 chars): {doctags[-200:]}")

            # v37: Collect for batch parsing
            all_doctags.append(doctags)
            all_images_for_parsing.append(batch_images[i])

    inference_time = time.time() - inference_start
    logger.info(f"[GraniteDocling] vLLM inference completed in {inference_time:.2f}s ({inference_time/len(rgb_images):.2f}s per page)")

    # v37: Parse ALL pages at once with docling-core
    # This is the critical fix - docling-core needs all pages together for cross-page table heuristics
    logger.info(f"[GraniteDocling] v37: Parsing all {len(all_doctags)} pages together with docling-core...")
    parse_start = time.time()
    result = parse_all_doctags_with_docling_core(all_doctags, all_images_for_parsing)
    parse_time = time.time() - parse_start
    logger.info(f"[GraniteDocling] docling-core parsing completed in {parse_time:.2f}s")

    return result, inference_time, len(rgb_images)


def handler(event):
    """
    RunPod serverless handler for Granite-Docling inference using direct vLLM.
    """
    start_time = time.time()

    try:
        input_data = event.get("input", {})
        pdf_base64 = input_data.get("pdf_base64")

        if not pdf_base64:
            return {"status": "error", "error": "No pdf_base64 provided"}

        # Process PDF with direct vLLM
        result, inference_time, page_count = process_pdf(pdf_base64)

        total_time = time.time() - start_time

        logger.info(f"[GraniteDocling] Processing completed:")
        logger.info(f"[GraniteDocling]   - Pages: {page_count}")
        logger.info(f"[GraniteDocling]   - Tables: {len(result['tables'])}")
        logger.info(f"[GraniteDocling]   - Text items: {len(result['text_content'])}")
        logger.info(f"[GraniteDocling]   - Inference time: {inference_time:.2f}s")
        logger.info(f"[GraniteDocling]   - Total time: {total_time:.2f}s")

        return {
            "status": "success",
            "result": {
                "markdown": result["markdown"],
                "tables": result["tables"],
                "text_content": result["text_content"],
                "metadata": {
                    "model": "granite-docling-258M",
                    "pipeline": "direct-vllm",
                    "framework": "vllm",
                    "accelerator": "cuda",
                    "inference_time_seconds": round(inference_time, 2),
                    "total_time_seconds": round(total_time, 2),
                    "table_count": len(result["tables"]),
                    "text_items": len(result["text_content"]),
                    "page_count": page_count
                }
            }
        }

    except Exception as e:
        logger.error(f"[GraniteDocling] Error: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    logger.info("[GraniteDocling] Starting RunPod serverless handler with direct vLLM...")
    logger.info("[GraniteDocling] Model: ibm-granite/granite-docling-258M (untied)")
    logger.info("[GraniteDocling] Approach: Direct vLLM inference (IBM production recommendation)")
    runpod.serverless.start({"handler": handler})
