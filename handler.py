"""
Granite-Docling-258M RunPod Serverless Handler (Direct Transformers Inference)

Runs IBM's Granite-Docling model via HuggingFace transformers (matching Docling's inline VLM pipeline)
and uses docling-core for DocTags -> Markdown conversion.

Key Features:
- Transformers-based Granite inference with untied weights (no vLLM dependency)
- 240 DPI PDF rendering streamed to /tmp (no long-lived PIL objects)
- docling-core multi-page parsing (DocTagsDocument + DoclingDocument) for cross-page tables
- Deterministic prompts/page-by-page generation with per-page markdown slices

API Input: {"input": {"pdf_base64": "base64_encoded_pdf"}}
API Output: {"status": "success", "result": {"markdown": "...", "tables": [...], "text_content": [...]} }

Reference:
- https://huggingface.co/ibm-granite/granite-docling-258M
- Docling inline Granite pipeline: https://docling-project.github.io/docling/

Build: 2025-12-20-v41-deprecation-fixes
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
import contextlib
from typing import List, Dict, Any, Tuple
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

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

# Granite model + processor globals
MODEL_NAME = "ibm-granite/granite-docling-258M"
MODEL_REVISION = "untied"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_model_dtype() -> torch.dtype:
    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


MODEL_DTYPE = _select_model_dtype()
model = None
processor = None


def load_transformer_model():
    """Load Granite-Docling through transformers (Docling inline pipeline)."""
    global model, processor

    if processor is None:
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True
        )

    if model is None:
        logger.info(
            "[GraniteDocling] Loading transformers model on %s (dtype=%s)...",
            device,
            MODEL_DTYPE,
        )
        dtype_arg = MODEL_DTYPE if device.type == "cuda" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=dtype_arg,  # Keep torch_dtype for now - dtype param not yet stable
        )
        model.to(device)
        model.eval()
        logger.info("[GraniteDocling] Model loaded successfully.")

    return model, processor


@contextlib.contextmanager
def autocast_if_available():
    """Enable CUDA autocast for transformer inference if a GPU is present."""
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=MODEL_DTYPE):
            yield
    else:
        yield


def generate_doctags_for_image(image: Image.Image, page_number: int) -> str:
    """Run Granite-Docling (transformers) on a single page image to obtain DocTags."""
    model, proc = load_transformer_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": DOC_TAG_PROMPT},
            ],
        }
    ]
    prompt = proc.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = proc(
        text=prompt,
        images=image,
        return_tensors="pt",
    )
    inputs = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]
    pad_token_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    eos_token_id = proc.tokenizer.eos_token_id

    generation_kwargs = dict(
        max_new_tokens=2048,
        do_sample=False,  # Greedy decoding (temperature not supported by this model)
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=True,
    )

    start = time.time()
    with torch.inference_mode():
        if device.type == "cuda":
            with autocast_if_available():
                output_ids = model.generate(**inputs, **generation_kwargs)
        else:
            output_ids = model.generate(**inputs, **generation_kwargs)
    elapsed = time.time() - start

    generated_ids = output_ids[:, input_len:]
    doctags_text = proc.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0].strip()

    if "</doctag>" not in doctags_text:
        doctags_text += "</doctag>"

    logger.info(
        "[GraniteDocling] Page %d: generated %d chars in %.2fs",
        page_number,
        len(doctags_text),
        elapsed,
    )
    return doctags_text


def render_pdf_to_disk(pdf_bytes: bytes) -> Tuple[List[str], str]:
    """
    Render PDF pages to temporary PNG files and return their paths.

    Storing the images on disk lets us stream them into the Granite transformer/docling-core without
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
    Process a PDF by streaming pages through the Granite-Docling transformers model
    and then parsing the combined DocTags with docling-core.
    """
    logger.info("[GraniteDocling] Decoding PDF payload...")
    pdf_bytes = base64.b64decode(pdf_base64)
    logger.info(f"[GraniteDocling] PDF size: {len(pdf_bytes)} bytes")

    image_paths, scratch_dir = render_pdf_to_disk(pdf_bytes)
    total_pages = len(image_paths)
    logger.info(f"[GraniteDocling] Processing {total_pages} pages with transformers Granite...")

    inference_start = time.time()
    all_doctags: List[str] = []

    for page_idx, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            doctags = generate_doctags_for_image(img, page_idx)
            all_doctags.append(doctags)

    inference_time = time.time() - inference_start
    logger.info(
        "[GraniteDocling] Inference completed in %.2fs (%.2fs per page)",
        inference_time,
        inference_time / max(total_pages, 1),
    )

    logger.info("[GraniteDocling] Parsing DocTags with docling-core (multi-page mode)...")
    parse_start = time.time()
    combined = parse_document_with_docling_core(all_doctags, image_paths)
    parse_time = time.time() - parse_start
    logger.info("[GraniteDocling] docling-core parsing completed in %.2fs", parse_time)

    shutil.rmtree(scratch_dir, ignore_errors=True)
    return combined, inference_time, total_pages


def handler(event):
    """
    RunPod serverless handler for Granite-Docling inference using transformers.
    """
    start_time = time.time()

    try:
        input_data = event.get("input", {})
        pdf_base64 = input_data.get("pdf_base64")

        if not pdf_base64:
            return {"status": "error", "error": "No pdf_base64 provided"}

        # Process PDF with the direct transformers pipeline
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
                    "pipeline": "direct-transformers",
                    "framework": "transformers",
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
    logger.info("[GraniteDocling] Starting RunPod serverless handler with transformers...")
    logger.info("[GraniteDocling] Model: ibm-granite/granite-docling-258M (untied)")
    logger.info("[GraniteDocling] Approach: Direct transformers inference (Docling inline pipeline)")
    runpod.serverless.start({"handler": handler})
