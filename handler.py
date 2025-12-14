"""
Granite-Docling-258M RunPod Serverless Handler

Uses Transformers for reliable inference with IBM's Granite-Docling model
for document understanding and table extraction.

Transformers is more reliable than VLLM for multimodal models.
Speed: ~100 tokens/sec (vs VLLM's ~500 tokens/sec) but guaranteed to work.

API Input: {"input": {"image_base64": "base64_encoded_image"}}
API Output: {"status": "success", "result": {"doctags": "...", "html": "...", "tables": [...]}}

Reference: https://huggingface.co/ibm-granite/granite-docling-258M
"""

import runpod
import base64
import io
import os
import re
import time
import traceback
import torch
from PIL import Image

# Global model and processor - loaded once, reused across requests
model = None
processor = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """
    Load Granite-Docling model with Transformers.

    Using flash_attention_2 on CUDA for better performance.
    """
    global model, processor

    if model is None:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        print(f"[GraniteDocling] Loading model with Transformers on {DEVICE}...")
        start_time = time.time()

        MODEL_PATH = "ibm-granite/granite-docling-258M"

        # Load processor
        processor = AutoProcessor.from_pretrained(MODEL_PATH)

        # Load model with optimal attention implementation
        attn_impl = "flash_attention_2" if DEVICE == "cuda" else "sdpa"
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            _attn_implementation=attn_impl,
        ).to(DEVICE)

        elapsed = time.time() - start_time
        print(f"[GraniteDocling] Model loaded in {elapsed:.2f}s")

    return model, processor


def extract_tables_from_html(html: str) -> list:
    """Extract table HTML segments from full document HTML."""
    if not html:
        return []

    tables = re.findall(r'<table[^>]*>.*?</table>', html, re.DOTALL | re.IGNORECASE)
    return [{"table_number": i + 1, "html": t} for i, t in enumerate(tables)]


def convert_doctags_to_html(doctags: str) -> str:
    """
    Convert DocTags format to HTML.

    DocTags is Granite-Docling's native output format.
    Uses docling-core library for conversion.
    """
    try:
        from docling_core.types.doc.document import DocTagsDocument

        # Parse DocTags and convert to HTML
        doc = DocTagsDocument.from_doctags(doctags)
        html = doc.export_to_html()
        return html
    except ImportError:
        print("[GraniteDocling] docling-core not available, returning raw doctags")
        return f"<pre>{doctags}</pre>"
    except Exception as e:
        print(f"[GraniteDocling] DocTags conversion error: {e}")
        return f"<pre>{doctags}</pre>"


def handler(event):
    """
    RunPod serverless handler for Granite-Docling inference.

    Input: {"input": {"image_base64": "..."}}
    Output: {"status": "success/error", "result": {...}}
    """
    start_time = time.time()

    try:
        input_data = event.get("input", {})
        image_base64 = input_data.get("image_base64")

        if not image_base64:
            return {"status": "error", "error": "No image_base64 provided"}

        # Decode image
        print("[GraniteDocling] Decoding image...")
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        print(f"[GraniteDocling] Image size: {image.size}")

        # Load model and processor
        mdl, proc = load_model()

        # Prepare input using chat template (as per HuggingFace docs)
        PROMPT_TEXT = "Convert this page to docling."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT_TEXT}
                ]
            }
        ]

        # Apply chat template and prepare inputs
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

        print("[GraniteDocling] Running inference...")
        inference_start = time.time()

        # Generate with Transformers
        with torch.no_grad():
            generated_ids = mdl.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=False,
            )

        inference_time = time.time() - inference_start
        print(f"[GraniteDocling] Inference completed in {inference_time:.2f}s")

        # Decode output
        # Remove input tokens from output
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        doctags = proc.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"[GraniteDocling] Output length: {len(doctags)} chars")

        # Convert DocTags to HTML
        print("[GraniteDocling] Converting DocTags to HTML...")
        html_output = convert_doctags_to_html(doctags)

        # Extract tables from HTML
        tables = extract_tables_from_html(html_output)
        print(f"[GraniteDocling] Found {len(tables)} tables")

        total_time = time.time() - start_time

        # Clear GPU cache to prevent memory buildup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "result": {
                "doctags": doctags,
                "html": html_output,
                "tables": tables,
                "metadata": {
                    "model": "granite-docling-258M",
                    "device": DEVICE,
                    "inference_time_seconds": round(inference_time, 2),
                    "total_time_seconds": round(total_time, 2),
                    "table_count": len(tables)
                }
            }
        }

    except Exception as e:
        print(f"[GraniteDocling] Error: {e}")
        traceback.print_exc()

        # Clear GPU cache on error
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Start RunPod serverless
if __name__ == "__main__":
    print("[GraniteDocling] Starting RunPod serverless handler...")
    print(f"[GraniteDocling] Device: {DEVICE}")
    print(f"[GraniteDocling] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[GraniteDocling] GPU: {torch.cuda.get_device_name(0)}")
    runpod.serverless.start({"handler": handler})
