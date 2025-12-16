"""
Granite-Docling-258M RunPod Serverless Handler (Direct vLLM Inference)

Uses direct vLLM API following IBM's recommended production approach.
No Docling SDK overhead - direct inference with custom DOCTAGS parsing.

Key Features:
- Direct vLLM.LLM() client with untied weights
- PDF rendered to RGB using pdf2image (no RGBA conversion)
- Custom DOCTAGS parser for table extraction
- 150 DPI rendering (matches IBM benchmarks)

API Input: {"input": {"pdf_base64": "base64_encoded_pdf"}}
API Output: {"status": "success", "result": {"markdown": "...", "tables": [...], "text_content": [...]}}

Reference:
- https://huggingface.co/ibm-granite/granite-docling-258M
- IBM recommendation: Direct vLLM for production (not Docling SDK)

Build: 2025-12-16-v28-sequential-pages
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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

        # Load vLLM model
        # Using bfloat16 for A4500/ADA GPUs which support it natively
        llm = LLM(
            model="ibm-granite/granite-docling-258M",
            revision="untied",  # CRITICAL - untied weights required
            limit_mm_per_prompt={"image": 1},  # CRITICAL - required for multimodal models
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=True,
            dtype="bfloat16"  # A4500/ADA GPUs support bfloat16
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

    # Convert PDF to images - specify dpi for quality
    images = convert_from_bytes(
        pdf_bytes,
        dpi=150,  # IBM benchmark tested at 150 DPI
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


def count_table_rows(markdown: str) -> int:
    """Count rows in markdown table (excluding header separator)."""
    if not markdown:
        return 0
    lines = markdown.strip().split('\n')
    return len([l for l in lines if l.strip().startswith('|') and '---' not in l])


def otsl_to_markdown(otsl_content: str) -> str:
    """
    Convert OTSL (Optimized Table Structure Language) format to markdown table.

    OTSL format: rows separated by newlines, cells by tabs
    """
    lines = otsl_content.strip().split('\n')

    if not lines:
        return ""

    # Build markdown table
    markdown_lines = []
    for i, line in enumerate(lines):
        cells = line.split('\t')
        markdown_lines.append('| ' + ' | '.join(cells) + ' |')

        # Add separator after header row
        if i == 0:
            markdown_lines.append('|' + '|'.join(['---' for _ in cells]) + '|')

    return '\n'.join(markdown_lines)


def parse_doctags(doctags_text: str) -> Dict[str, Any]:
    """
    Parse DOCTAGS format into structured data.

    DOCTAGS format:
    <otsl>
      table_content_here
    </otsl>
    <text>
      text_content_here
    </text>
    """
    tables = []
    text_content = []

    # Find all table tags
    table_pattern = r'<otsl>(.*?)</otsl>'
    table_matches = re.findall(table_pattern, doctags_text, re.DOTALL)

    logger.info(f"[GraniteDocling] Found {len(table_matches)} OTSL tables in DOCTAGS")

    for i, table_content in enumerate(table_matches):
        # Convert OTSL to markdown
        markdown = otsl_to_markdown(table_content)
        row_count = count_table_rows(markdown)

        logger.info(f"[GraniteDocling] Table {i+1}: {row_count} rows")

        tables.append({
            "table_number": i + 1,
            "markdown": markdown,
            "row_count": row_count
        })

    # Find text content (everything not in tables)
    text_pattern = r'<text>(.*?)</text>'
    text_matches = re.findall(text_pattern, doctags_text, re.DOTALL)

    for text in text_matches:
        text_content.append({
            "text": text.strip(),
            "type": "text"
        })

    return {
        "tables": tables,
        "text_content": text_content,
        "raw_doctags": doctags_text
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

        # Renumber tables globally
        for table in page_result.get("tables", []):
            table_counter += 1
            table["table_number"] = table_counter
            all_tables.append(table)
            full_markdown.append(table.get("markdown", ""))

        # Add text content
        for text_item in page_result.get("text_content", []):
            all_text.append(text_item)
            full_markdown.append(text_item.get("text", ""))

    return {
        "tables": all_tables,
        "text_content": all_text,
        "markdown": '\n\n'.join(full_markdown),
        "raw_doctags": [p.get("raw_doctags", "") for p in page_results]
    }


def process_pdf(pdf_base64: str) -> Dict[str, Any]:
    """
    Process PDF with direct vLLM inference using SEQUENTIAL page processing.

    Steps:
    1. Decode PDF bytes
    2. Render to RGB images (150 DPI, PPM format)
    3. Process each page SEQUENTIALLY (not batched) to avoid encoder cache exhaustion
    4. Parse DOCTAGS output for each page
    5. Combine pages

    NOTE: Sequential processing is required because vLLM's encoder cache (8192 tokens)
    gets exhausted when processing multiple images in a single batch, causing
    subsequent pages to produce empty OTSL output.
    See: https://github.com/vllm-project/vllm/issues/20123
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
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=8192,  # IBM's recommended value for full DOCTAGS output
        skip_special_tokens=False  # Preserve DOCTAGS
    )

    # Process pages SEQUENTIALLY to avoid encoder cache exhaustion
    # Each generate() call gets a fresh encoder cache
    logger.info(f"[GraniteDocling] Processing {len(rgb_images)} pages SEQUENTIALLY...")
    inference_start = time.time()
    results = []

    for i, image in enumerate(rgb_images):
        page_start = time.time()

        # Use processor.apply_chat_template as per IBM documentation
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)

        # DEBUG: Log the prompt being used (only first page to avoid spam)
        if i == 0:
            logger.info(f"[GraniteDocling] Using prompt format: {prompt[:200]}...")

        # Create single-page prompt
        single_prompt = [{
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }]

        # Process single page - fresh encoder cache for each page
        logger.info(f"[GraniteDocling] Processing page {i+1}/{len(rgb_images)}...")
        page_output = llm_client.generate(single_prompt, sampling_params=sampling_params)

        # Extract DOCTAGS from output
        doctags = page_output[0].outputs[0].text
        page_time = time.time() - page_start

        logger.info(f"[GraniteDocling] Page {i+1} completed in {page_time:.2f}s: {len(doctags)} chars")
        # DEBUG: Log first 500 chars of raw output to see actual format
        logger.info(f"[GraniteDocling] Page {i+1} raw output (first 500 chars): {doctags[:500]}")
        # DEBUG: Log last 200 chars to check if output is truncated
        logger.info(f"[GraniteDocling] Page {i+1} raw output (last 200 chars): {doctags[-200:]}")

        # Parse DOCTAGS for this page
        parsed = parse_doctags(doctags)
        results.append(parsed)

    inference_time = time.time() - inference_start
    logger.info(f"[GraniteDocling] All pages completed in {inference_time:.2f}s ({inference_time/len(rgb_images):.2f}s per page)")

    # Combine pages
    combined = combine_pages(results)

    return combined, inference_time, len(rgb_images)


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
