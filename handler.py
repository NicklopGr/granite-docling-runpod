"""
Granite-Docling-258M RunPod Serverless Handler (Direct vLLM Inference)

Uses direct vLLM API following IBM's recommended production approach.
Uses docling-core for proper DocTags -> Markdown conversion.

Key Features:
- Direct vLLM.LLM() client with untied weights
- 240 DPI PDF rendering streamed to /tmp (no long-lived PIL objects)
- docling-core multi-page parsing (DocTagsDocument + DoclingDocument) for cross-page tables
- Single-page vLLM batches (matches transformers.js demo) to keep prompts deterministic
- v39: Restored continuation heuristics + page-specific markdown slices

API Input: {"input": {"pdf_base64": "base64_encoded_pdf"}}
API Output: {"status": "success", "result": {"markdown": "...", "tables": [...], "text_content": [...]}}

Reference:
- https://huggingface.co/ibm-granite/granite-docling-258M
- IBM recommendation: Direct vLLM for production (not Docling SDK)

Build: 2025-12-18-v39-streaming-multipage
"""

import runpod
import base64
import os
import time
import traceback
import json
import logging
import re
import shutil
import tempfile
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Prompt reminding Granite about DocTags schema (mirrors IBM reference)
DOC_TAG_PROMPT = """You are Docling, an AI that converts document images into DocTags.
Follow this schema exactly:
1. Begin with <doctag> and end with </doctag>.
2. Emit layout tags such as <section_header_level_1>, <text>, <picture>, <otsl>, <ched>, <fcel>, <ecel>.
3. For every table (including continuation pages):
   - Output a single <otsl> enclosing the table coordinates.
   - Produce header rows with <ched>.
   - Produce every table cell with <fcel> ... </fcel> in strict row-major order.
   - Keep textual descriptions inside the Description column. Do NOT move description text into the debit/credit cells; numeric values must stay under the debit/credit headers.
   - Each <fcel> must contain the textual value for that cell. Never leave a cell empty and never omit debit/credit columns.
   - Continuation pages must restate all columns (date, description, debit, credit, balance) exactly like the first page.
4. If a source cell is blank, still emit <fcel></fcel> so downstream parsers know it was intentionally empty.
5. Preserve reading order from top-left to bottom-right and finish only after closing </doctag>.
Convert the provided page image into precise DocTags with complete table markup."""

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
            limit_mm_per_prompt={"image": 4},  # Allow multiple images prompt context
            trust_remote_code=True,
            enable_prefix_caching=False  # Prevent cache reuse across different PDFs
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


def render_pdf_to_disk(pdf_bytes: bytes) -> Tuple[List[str], str]:
    """
    Render PDF pages to temporary PNG files and return their paths.

    Storing the images on disk lets us stream them into vLLM/docling-core without
    keeping dozens of PIL objects resident in GPU worker RAM.
    """
    from pdf2image import convert_from_bytes

    scratch_dir = tempfile.mkdtemp(prefix="granite_pages_")
    logger.info(f"[GraniteDocling] Rendering PDF to {scratch_dir} at 240 DPI...")
    start_time = time.time()

    image_paths = convert_from_bytes(
        pdf_bytes,
        dpi=240,
        fmt="png",
        output_folder=scratch_dir,
        paths_only=True,
    )

    # pdf2image preserves page order, but sort defensively.
    image_paths = sorted(image_paths)
    elapsed = time.time() - start_time
    logger.info(f"[GraniteDocling] Rendered {len(image_paths)} pages in {elapsed:.2f}s")

    return image_paths, scratch_dir


def load_image_as_rgb(image_path: str) -> Image.Image:
    """Load an image file as RGB, returning a new PIL object the caller must close."""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img.copy()


def split_markdown_into_pages(markdown: str, total_pages: int) -> List[str]:
    """
    Split combined markdown into per-page segments using Docling's
    "--- Page N ---" delimiters. Pads/truncates to total_pages entries.
    """
    if not markdown:
        return ["" for _ in range(total_pages)]

    pattern = re.compile(r"\n+--- Page (\d+) ---\n+")
    sections: List[str] = []
    last_idx = 0

    for match in pattern.finditer(markdown):
        sections.append(markdown[last_idx:match.start()].strip())
        last_idx = match.end()

    sections.append(markdown[last_idx:].strip())

    if len(sections) < total_pages:
        sections.extend([""] * (total_pages - len(sections)))
    elif len(sections) > total_pages:
        sections = sections[:total_pages]

    return sections


def parse_document_with_docling_core(
    doctags_list: List[str],
    image_paths: List[str]
) -> Dict[str, Any]:
    """
    Parse ALL DocTags at once using docling-core so continuation tables retain context.
    """
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument

    logger.info(
        f"[GraniteDocling] Multi-page docling-core parse starting "
        f"(pages={len(doctags_list)})"
    )

    images: List[Image.Image] = []
    try:
        for path in image_paths:
            images.append(load_image_as_rgb(path))

        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            doctags_list,
            images,
        )
        doc = DoclingDocument.load_from_doctags(
            doctags_doc,
            document_name="GraniteDoclingDocument",
        )

        full_markdown = doc.export_to_markdown()
        page_markdowns = split_markdown_into_pages(
            full_markdown,
            total_pages=len(image_paths),
        )

        tables = []
        if hasattr(doc, "tables") and doc.tables:
            logger.info(f"[GraniteDocling] docling-core surfaced {len(doc.tables)} tables")
            for idx, table in enumerate(doc.tables):
                try:
                    table_md = ""
                    if hasattr(table, "export_to_markdown"):
                        try:
                            table_md = table.export_to_markdown(doc=doc)
                        except TypeError:
                            table_md = table.export_to_markdown()
                    else:
                        table_md = str(table)

                    row_count = (
                        len(table_md.strip().split("\n")) - 1 if table_md.strip() else 0
                    )
                    page_number = None
                    if getattr(table, "prov", None):
                        prov = table.prov[0]
                        page_number = (prov.page_no or 0) + 1

                    tables.append(
                        {
                            "table_number": idx + 1,
                            "markdown": table_md,
                            "row_count": max(row_count, 0),
                            "page_number": page_number,
                        }
                    )
                except Exception as table_error:
                    logger.exception(
                        f"[GraniteDocling] FAILED exporting table {idx + 1}: {table_error}"
                    )
                    tables.append(
                        {
                            "table_number": idx + 1,
                            "markdown": "",
                            "row_count": 0,
                            "page_number": None,
                            "error": str(table_error),
                        }
                    )

        text_content = []
        if hasattr(doc, "texts") and doc.texts:
            for text in doc.texts:
                page_number = None
                if getattr(text, "prov", None):
                    page_number = (text.prov[0].page_no or 0) + 1
                text_content.append(
                    {
                        "text": str(text),
                        "type": "text",
                        "page_number": page_number,
                    }
                )

        return {
            "tables": tables,
            "text_content": text_content,
            "markdown": full_markdown,
            "page_markdowns": page_markdowns,
            "raw_doctags": doctags_list,
        }

    except Exception as e:
        logger.exception("[GraniteDocling] docling-core multi-page parsing failed: %s", e)
        raise
    finally:
        for img in images:
            try:
                img.close()
            except Exception:
                pass


def process_pdf(pdf_base64: str) -> Dict[str, Any]:
    """
    Process PDF with direct vLLM inference using BATCH processing.

    Steps:
    1. Decode PDF bytes
    2. Render images to disk at 240 DPI (streamed RGB PNGs)
    3. Process ALL pages via single-page llm.generate() calls
    4. Collect ALL DocTags, then parse with docling-core ONCE (multi-page mode)
    5. Return combined result with per-page markdown slices

    v31: Switched from sequential to batch processing.
    v39: Parse all pages at once with docling-core for cross-page table context.
    See: https://github.com/docling-project/docling/blob/main/docling/models/vlm_models_inline/vllm_model.py
    """
    from vllm import SamplingParams

    # Load vLLM client
    llm_client, proc = load_vllm()

    # Decode PDF
    logger.info("[GraniteDocling] Decoding PDF...")
    pdf_bytes = base64.b64decode(pdf_base64)
    logger.info(f"[GraniteDocling] PDF size: {len(pdf_bytes)} bytes")

    # Render to disk
    image_paths, scratch_dir = render_pdf_to_disk(pdf_bytes)

    # IBM's official message format for granite-docling
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": DOC_TAG_PROMPT}
            ],
        },
    ]

    # Configure sampling
    # v33: Added stop strings from Docling's vlm_model_specs.py
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=10000,  # Extra headroom to finish DocTags on continuation pages
        repetition_penalty=1.05,  # Encourage Granite to keep emitting cell content
        skip_special_tokens=False,  # Preserve DOCTAGS
        stop=["</doctag>", "<|end_of_text|>"]  # v33: Docling's stop strings
    )

    # Force single-page batches (matches transformers.js demo) to avoid continuation loss
    MAX_PAGES_PER_BATCH = 1

    total_pages = len(image_paths)
    logger.info(
        f"[GraniteDocling] Processing {total_pages} pages "
        f"(max {MAX_PAGES_PER_BATCH} per batch)..."
    )
    inference_start = time.time()

    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    logger.info(f"[GraniteDocling] Using prompt format: {prompt[:200]}...")

    # Collect all doctags for multi-page parsing
    all_doctags: List[str] = []

    try:
        # Process pages in chunks of MAX_PAGES_PER_BATCH (vLLM inference)
        for batch_start in range(0, total_pages, MAX_PAGES_PER_BATCH):
            batch_end = min(batch_start + MAX_PAGES_PER_BATCH, total_pages)
            batch_paths = image_paths[batch_start:batch_end]
            batch_num = (batch_start // MAX_PAGES_PER_BATCH) + 1
            total_batches = (total_pages + MAX_PAGES_PER_BATCH - 1) // MAX_PAGES_PER_BATCH

            logger.info(
                f"[GraniteDocling] Batch {batch_num}/{total_batches}: "
                f"Pages {batch_start + 1}-{batch_end}"
            )

            # Build batch inputs for this chunk
            batched_inputs = []
            opened_images: List[Image.Image] = []
            for i, image_path in enumerate(batch_paths):
                image = load_image_as_rgb(image_path)
                opened_images.append(image)
                batched_inputs.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image},
                    }
                )
                logger.info(f"[GraniteDocling] Added page {batch_start + i + 1} to batch")

            # Single generate call for this batch
            logger.info(
                f"[GraniteDocling] Running batch inference for {len(batched_inputs)} pages..."
            )
            outputs = llm_client.generate(
                batched_inputs,
                sampling_params=sampling_params,
            )

            # Close PIL images immediately after inference
            for img in opened_images:
                try:
                    img.close()
                except Exception:
                    pass

            # Collect outputs for downstream parsing
            for i, output in enumerate(outputs):
                page_num = batch_start + i + 1
                doctags = output.outputs[0].text or ""

                logger.info(f"[GraniteDocling] Page {page_num}: {len(doctags)} chars")
                logger.info(
                    f"[GraniteDocling] Page {page_num} raw output (first 500 chars): "
                    f"{doctags[:500]}"
                )
                logger.info(
                    f"[GraniteDocling] Page {page_num} raw output (last 200 chars): "
                    f"{doctags[-200:]}"
                )

                all_doctags.append(doctags)

        inference_time = time.time() - inference_start
        per_page_time = inference_time / total_pages if total_pages else 0
        logger.info(
            f"[GraniteDocling] vLLM inference completed in {inference_time:.2f}s "
            f"({per_page_time:.2f}s per page)"
        )

        if len(all_doctags) != total_pages:
            logger.warning(
                "[GraniteDocling] Page count mismatch (doctags=%s, pages=%s). "
                "Padding with empty DocTags.",
                len(all_doctags),
                total_pages,
            )
            while len(all_doctags) < total_pages:
                all_doctags.append("")

        logger.info(
            f"[GraniteDocling] docling-core multi-page parse kicking off "
            f"(pages={len(all_doctags)})"
        )
        parse_start = time.time()
        combined = parse_document_with_docling_core(all_doctags, image_paths)
        parse_time = time.time() - parse_start
        logger.info(f"[GraniteDocling] docling-core parsing completed in {parse_time:.2f}s")

        return combined, inference_time, total_pages

    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


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
                "page_markdowns": result.get("page_markdowns", []),
                "tables": result["tables"],
                "text_content": result["text_content"],
                "raw_doctags": result.get("raw_doctags", []),
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
