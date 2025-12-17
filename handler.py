"""
Granite-Docling-258M RunPod Serverless Handler (Direct vLLM Inference)

Uses direct vLLM API following IBM's recommended production approach.
Uses docling-core for proper DocTags -> Markdown conversion plus PaddleOCR-VL fallback for continuation tables.

Key Features:
- Direct vLLM.LLM() client with untied weights
- PDF rendered to RGB using pdf2image (no RGBA conversion)
- docling-core for DocTags parsing (DocTagsDocument + DoclingDocument)
- 192 DPI rendering for sharper table detection
- Max 10 pages per vLLM batch (memory optimization)
- Paged docling-core parsing + PaddleOCR-VL fallback for empty tables

API Input: {"input": {"pdf_base64": "base64_encoded_pdf"}}
API Output: {"status": "success", "result": {"markdown": "...", "tables": [...], "text_content": [...]}}

Reference:
- https://huggingface.co/ibm-granite/granite-docling-258M
- IBM recommendation: Direct vLLM for production (not Docling SDK)

Build: 2025-12-17-v39-paddle-fallback
"""

import runpod
import base64
import io
import os
import time
import traceback
import json
import logging
import re
import html
from html.parser import HTMLParser
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torch
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Prompt reminding Granite about DocTags schema (mirrors IBM reference)
DOC_TAG_PROMPT = """You are Docling, an AI that converts document images into DocTags.
Follow this schema exactly:
- Begin with <doctag> and end with </doctag>.
- Use structural tags such as <section_header_level_1>, <text>, <picture>, <otsl>, <ched>, <fcel>, <ecel>.
- For tables, emit <otsl> for the table region and then DocTags rows using <ched> (header), <fcel> (cell), <ecel>.
- Preserve the reading order from top-left to bottom-right.
Convert the provided page image into DocTags with accurate table markup."""

# Optional PaddleOCR-VL fallback (cropped table recovery)
PADDLE_VL_ENDPOINT_ID = os.getenv("PADDLE_VL_ENDPOINT_ID")
PADDLE_VL_API_KEY = os.getenv("PADDLE_VL_API_KEY") or os.getenv("RUNPOD_API_KEY")
PADDLE_VL_TIMEOUT = int(os.getenv("PADDLE_VL_TIMEOUT", "60"))
PADDLE_VL_PADDING = int(os.getenv("PADDLE_VL_PADDING", "16"))

# DocTags coordinate helpers
MAX_DOCTAG_COORD = 512.0
OTS_REGION_PATTERN = re.compile(r"<otsl>(.*?)</otsl>", re.DOTALL)
LOC_PATTERN = re.compile(r"<loc_(\d+)>")


class SimpleHTMLTableParser(HTMLParser):
    """Minimal HTML table parser to convert Paddle outputs into Markdown."""

    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._current_row: List[str] = []
        self._current_cell: List[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = []
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            text = html.unescape(''.join(self._current_cell).strip())
            self._current_row.append(text)
            self._current_cell = []
            self._in_cell = False
        elif tag == "tr":
            if any(cell.strip() for cell in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell.append(data)


# v35: Enable TensorFloat32 tensor cores for faster float32 matrix multiplication
# This addresses the warning: "TensorFloat32 tensor cores available but not enabled"
# TF32 uses 19-bit precision (vs 32-bit) for 8x faster matmul with minimal accuracy loss
torch.set_float32_matmul_precision('high')
logger.info("[GraniteDocling] TF32 tensor cores enabled (float32_matmul_precision='high')")

# Global vLLM client - loaded once, reused across requests
llm = None


def load_vllm():
    """Load vLLM with Granite-Docling-258M (untied weights)."""
    global llm

    if llm is None:
        from vllm import LLM

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
            limit_mm_per_prompt={"image": 4},  # Allow multiple images prompt context
            trust_remote_code=True,
            enable_prefix_caching=True  # Align with docling inline VLM defaults
            # Let vLLM auto-detect: dtype, gpu_memory_utilization (0.9), max_model_len
        )

        elapsed = time.time() - start_time
        logger.info(f"[GraniteDocling] vLLM loaded in {elapsed:.2f}s")

    return llm


def render_pdf_to_rgb(pdf_bytes: bytes) -> List[Image.Image]:
    """
    Render PDF pages directly to RGB (no RGBA conversion needed).

    Uses pdf2image with PPM format which produces RGB by default.
    """
    from pdf2image import convert_from_bytes

    logger.info("[GraniteDocling] Rendering PDF to RGB images...")
    start_time = time.time()

    # Convert PDF to images
    # Increase to 192 DPI for sharper tables (helps Granite detect gridlines)
    images = convert_from_bytes(
        pdf_bytes,
        dpi=192,
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


def extract_table_regions_from_doctags(
    doctags_text: str,
    width: int,
    height: int
) -> List[Dict[str, float]]:
    """Parse <otsl> regions from DocTags and map them to pixel coordinates."""
    regions: List[Dict[str, float]] = []
    if not doctags_text:
        return regions

    for match in OTS_REGION_PATTERN.finditer(doctags_text):
        locs = LOC_PATTERN.findall(match.group(1))
        if len(locs) < 4:
            continue
        coords = [max(0.0, min(MAX_DOCTAG_COORD, float(value))) for value in locs[:4]]
        x1, y1, x2, y2 = coords
        regions.append({
            "x1": (x1 / MAX_DOCTAG_COORD) * width,
            "y1": (y1 / MAX_DOCTAG_COORD) * height,
            "x2": (x2 / MAX_DOCTAG_COORD) * width,
            "y2": (y2 / MAX_DOCTAG_COORD) * height,
        })

    return regions


def table_html_to_markdown(html_table: str) -> Tuple[str, int]:
    """Convert Paddle's HTML table snippet into Markdown."""
    parser = SimpleHTMLTableParser()
    parser.feed(html_table or "")
    parser.close()

    if not parser.rows:
        return "", 0

    col_count = max(len(row) for row in parser.rows)
    normalized_rows = [
        row + [""] * (col_count - len(row))
        for row in parser.rows
    ]

    header = normalized_rows[0]
    body = normalized_rows[1:] if len(normalized_rows) > 1 else []
    header_line = "| " + " | ".join(header) + " |"
    separator_line = "| " + " | ".join(["---"] * col_count) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in body]

    markdown = "\n".join([header_line, separator_line] + body_lines)
    return markdown.strip(), len(body)


def crop_image_to_bbox(
    image: Image.Image,
    bbox: Dict[str, float],
    padding: int
) -> Optional[Image.Image]:
    """Crop PIL image to bounding box with optional padding."""
    width, height = image.size
    x1 = max(0, int(bbox.get("x1", 0)) - padding)
    y1 = max(0, int(bbox.get("y1", 0)) - padding)
    x2 = min(width, int(bbox.get("x2", width)) + padding)
    y2 = min(height, int(bbox.get("y2", height)) + padding)

    if x1 >= x2 or y1 >= y2:
        return None

    return image.crop((x1, y1, x2, y2))


def image_to_base64(image: Image.Image) -> str:
    """Encode PIL image as base64 PNG."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def call_paddle_vl(image_base64: str) -> Optional[Dict[str, Any]]:
    """Invoke PaddleOCR-VL RunPod endpoint synchronously."""
    if not (PADDLE_VL_ENDPOINT_ID and PADDLE_VL_API_KEY):
        return None

    url = f"https://api.runpod.ai/v2/{PADDLE_VL_ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {PADDLE_VL_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json={"input": {"image_base64": image_base64}},
            timeout=PADDLE_VL_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as err:
        logger.warning(f"[GraniteDocling] PaddleOCR-VL request failed: {err}")
        return None

    output = payload.get("output") or {}
    result = output.get("result") or output
    if isinstance(result, dict) and "pages" in result:
        pages = result.get("pages")
    else:
        pages = None

    if not pages and isinstance(output, dict) and "pages" in output:
        pages = output.get("pages")

    if pages and isinstance(pages, list) and pages:
        return pages[0]

    logger.warning("[GraniteDocling] PaddleOCR-VL response did not include pages")
    return None


def extract_table_from_paddle_page(page_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the first table from Paddle OCR raw output."""
    raw = page_data.get("raw", {})
    parsing_list = raw.get("res", {}).get("parsing_res_list", [])

    candidates = []
    if isinstance(parsing_list, list):
        for block in parsing_list:
            if (
                isinstance(block, dict)
                and block.get("block_label") == "table"
                and block.get("block_content")
            ):
                markdown, row_count = table_html_to_markdown(block["block_content"])
                if markdown:
                    candidates.append({"markdown": markdown, "row_count": row_count})

    if not candidates:
        blocks = page_data.get("blocks", [])
        if isinstance(blocks, list):
            for block in blocks:
                if (
                    isinstance(block, dict)
                    and block.get("block_label") == "table"
                    and block.get("block_content")
                ):
                    markdown, row_count = table_html_to_markdown(block["block_content"])
                    if markdown:
                        candidates.append({"markdown": markdown, "row_count": row_count})
                        break

    return candidates[0] if candidates else None


def recover_table_with_paddle(
    image: Image.Image,
    bbox: Dict[str, float],
    page_number: int,
    table_number: int
) -> Optional[Dict[str, Any]]:
    """Crop table region and attempt to recover rows using PaddleOCR-VL."""
    cropped = crop_image_to_bbox(image, bbox, PADDLE_VL_PADDING)
    if cropped is None:
        logger.warning(f"[GraniteDocling] Cannot crop bbox for page {page_number}, table {table_number}")
        return None

    paddle_page = call_paddle_vl(image_to_base64(cropped))
    if not paddle_page:
        return None

    table_data = extract_table_from_paddle_page(paddle_page)
    if table_data:
        logger.info(
            f"[GraniteDocling] Paddle fallback recovered page {page_number} table {table_number} "
            f"with {table_data['row_count']} rows"
        )
        return table_data

    logger.warning(f"[GraniteDocling] Paddle fallback returned no tables for page {page_number}, table {table_number}")
    return None


def apply_paddle_fallback(
    page_results: List[Dict[str, Any]],
    page_images: List[Image.Image]
) -> int:
    """Run Paddle fallback for tables with zero rows, returns recovered count."""
    if not (PADDLE_VL_ENDPOINT_ID and PADDLE_VL_API_KEY):
        logger.info("[GraniteDocling] Paddle fallback not configured; skipping")
        return 0

    recovered = 0
    for idx, page in enumerate(page_results):
        image = page_images[idx]
        for table in page.get("tables", []):
            if table.get("row_count", 0) > 0:
                continue
            bbox = table.get("bbox")
            if not bbox:
                logger.warning(
                    f"[GraniteDocling] No bbox for page {page.get('page_number')} table {table.get('table_number')}, "
                    "cannot run Paddle fallback"
                )
                continue

            fallback_data = recover_table_with_paddle(
                image,
                bbox,
                page.get("page_number", idx + 1),
                table.get("table_number", 0)
            )
            if fallback_data and fallback_data.get("markdown"):
                table["markdown"] = fallback_data["markdown"]
                table["row_count"] = fallback_data["row_count"]
                table["source"] = "paddle-ocr-vl"
                table["recovered"] = True
                page_markdown = page.get("markdown") or ""
                addition = f"\n\n{fallback_data['markdown']}"
                page["markdown"] = (page_markdown + addition).strip() if page_markdown else fallback_data["markdown"]
                recovered += 1

    return recovered


def parse_page_with_docling_core(
    doctags_text: str,
    image: Image.Image,
    page_number: int,
    table_regions: List[Dict[str, float]]
) -> Dict[str, Any]:
    """
    Parse a single page of DocTags using docling-core.
    This mirrors the transformers.js demo (one image at a time) and avoids
    the current limitations docling-core has with multi-page continuation tables.
    """
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument

    logger.info(f"[GraniteDocling] v39: Parsing page {page_number} with docling-core")

    try:
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags_text], [image])
        doc = DoclingDocument.load_from_doctags(
            doctags_doc,
            document_name=f"Page {page_number}"
        )

        markdown = doc.export_to_markdown()
        tables = []
        text_content = []

        if hasattr(doc, 'tables') and doc.tables:
            logger.info(f"[GraniteDocling] v39: Found {len(doc.tables)} tables on page {page_number}")
            for i, table in enumerate(doc.tables):
                try:
                    logger.info(f"[GraniteDocling] v39: Processing table {i + 1}/{len(doc.tables)} on page {page_number}")

                    table_md = ""
                    if hasattr(table, 'export_to_markdown'):
                        try:
                            table_md = table.export_to_markdown(doc=doc)
                        except TypeError as te:
                            logger.warning(f"[GraniteDocling] v39: Table {i+1} export_to_markdown(doc=doc) failed on page {page_number}: {te}")
                            table_md = table.export_to_markdown()
                    else:
                        table_md = str(table)

                    row_count = len(table_md.strip().split('\n')) - 1 if table_md else 0
                    table_info = {
                        "table_number": i + 1,
                        "markdown": table_md,
                        "row_count": row_count,
                        "page_number": page_number
                    }
                    if i < len(table_regions):
                        table_info["bbox"] = table_regions[i]
                    tables.append(table_info)
                    logger.info(f"[GraniteDocling] Table {i+1} on page {page_number}: {row_count} rows")
                    if row_count == 0 or not table_md.strip():
                        logger.warning(f"[GraniteDocling] v39: Table {i+1} markdown empty on page {page_number} (likely continuation)")

                except Exception as table_error:
                    logger.exception(f"[GraniteDocling] v39: FAILED processing table {i+1} on page {page_number}: {table_error}")
                    tables.append({
                        "table_number": i + 1,
                        "markdown": "",
                        "row_count": 0,
                        "page_number": page_number,
                        "error": str(table_error)
                    })

        if hasattr(doc, 'texts') and doc.texts:
            for text in doc.texts:
                text_content.append({
                    "text": str(text),
                    "type": "text",
                    "page_number": page_number
                })

        logger.info(f"[GraniteDocling] Page {page_number}: docling-core parsed {len(tables)} tables, {len(text_content)} text items")
        logger.info(f"[GraniteDocling] Page {page_number}: Markdown length {len(markdown)} chars")

        return {
            "tables": tables,
            "text_content": text_content,
            "markdown": markdown,
            "raw_doctags": doctags_text,
            "page_number": page_number,
            "table_regions": table_regions,
            "image_size": image.size
        }

    except Exception as e:
        logger.exception(f"[GraniteDocling] docling-core parsing failed on page {page_number}: {e}")
        return {
            "tables": [],
            "text_content": [{"text": doctags_text, "type": "raw_doctags", "page_number": page_number}],
            "markdown": doctags_text,
            "raw_doctags": doctags_text,
            "page_number": page_number,
            "error": str(e),
            "table_regions": table_regions,
            "image_size": image.size
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
    llm_client = load_vllm()

    # Decode PDF
    logger.info("[GraniteDocling] Decoding PDF...")
    pdf_bytes = base64.b64decode(pdf_base64)
    logger.info(f"[GraniteDocling] PDF size: {len(pdf_bytes)} bytes")

    # Render to RGB
    rgb_images = render_pdf_to_rgb(pdf_bytes)

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

    prompt = f"<|start_of_role|>user<|end_of_role|><image>\n{DOC_TAG_PROMPT.strip()}\n<|end_of_text|>"
    logger.info(f"[GraniteDocling] Using prompt format: {prompt[:200]}...")

    # Collect all doctags/images first, then parse page-by-page
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

    # Parse each page independently (matches transformers.js demo behavior)
    logger.info(f"[GraniteDocling] v39: Parsing {len(all_doctags)} pages individually with docling-core...")
    parse_start = time.time()
    page_results = []
    for page_idx, (doctags, image) in enumerate(zip(all_doctags, all_images_for_parsing), start=1):
        table_regions = extract_table_regions_from_doctags(doctags, image.width, image.height)
        page_results.append(parse_page_with_docling_core(doctags, image, page_idx, table_regions))

    parse_time = time.time() - parse_start
    logger.info(f"[GraniteDocling] docling-core page-by-page parsing completed in {parse_time:.2f}s")

    empty_tables_before = sum(
        1
        for page in page_results
        for table in page.get("tables", [])
        if table.get("row_count", 0) == 0
    )
    paddle_recovered = apply_paddle_fallback(page_results, all_images_for_parsing)
    if paddle_recovered:
        logger.info(f"[GraniteDocling] Paddle fallback recovered {paddle_recovered} table(s)")

    combined = combine_pages(page_results)
    combined["fallback_summary"] = {
        "empty_tables_before_fallback": empty_tables_before,
        "paddle_recovered_tables": paddle_recovered
    }
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

        fallback_summary = result.get("fallback_summary") or {}
        response_payload = {
            "status": "success",
            "result": {
                "markdown": result["markdown"],
                "tables": result["tables"],
                "text_content": result["text_content"],
                "raw_doctags": result.get("raw_doctags"),
                "fallback_summary": fallback_summary,
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
        if fallback_summary:
            response_payload["result"]["metadata"]["fallback"] = fallback_summary

        return response_payload

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
