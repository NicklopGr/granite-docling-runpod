"""
Granite-Docling-258M RunPod Serverless Handler

Uses VLLM for fast inference with IBM's Granite-Docling model
for document understanding and table extraction.

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
from PIL import Image

# Global model - loaded once, reused across requests
llm = None


def load_model():
    """
    Load Granite-Docling model with VLLM.
    """
    global llm

    if llm is None:
        from vllm import LLM

        print("[GraniteDocling] Loading model with VLLM...")
        start_time = time.time()

        MODEL_PATH = "ibm-granite/granite-docling-258M"

        llm = LLM(
            model=MODEL_PATH,
            revision="untied",
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            limit_mm_per_prompt={"image": 1},
        )

        elapsed = time.time() - start_time
        print(f"[GraniteDocling] Model loaded in {elapsed:.2f}s")

    return llm


def extract_tables_from_html(html: str) -> list:
    """Extract table HTML segments from full document HTML."""
    if not html:
        return []

    tables = re.findall(r'<table[^>]*>.*?</table>', html, re.DOTALL | re.IGNORECASE)
    return [{"table_number": i + 1, "html": t} for i, t in enumerate(tables)]


def convert_doctags_to_html(doctags: str) -> str:
    """
    Convert DocTags format to HTML.
    """
    try:
        from docling_core.types.doc.document import DocTagsDocument
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

        # Load model
        model = load_model()

        from vllm import SamplingParams

        # Simple prompt with single image placeholder
        # Using format similar to other VLMs that work with VLLM
        prompt = "User:<image>Convert this page to docling.\nAssistant:"

        sampling_params = SamplingParams(
            max_tokens=8192,
            temperature=0,
        )

        print(f"[GraniteDocling] Prompt: {prompt}")
        print("[GraniteDocling] Running inference...")
        inference_start = time.time()

        # Pass PIL image directly
        outputs = model.generate(
            [{
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            }],
            sampling_params=sampling_params
        )

        inference_time = time.time() - inference_start
        print(f"[GraniteDocling] Inference completed in {inference_time:.2f}s")

        # Extract DocTags output
        doctags = outputs[0].outputs[0].text
        print(f"[GraniteDocling] Output length: {len(doctags)} chars")

        # Convert DocTags to HTML
        print("[GraniteDocling] Converting DocTags to HTML...")
        html_output = convert_doctags_to_html(doctags)

        # Extract tables from HTML
        tables = extract_tables_from_html(html_output)
        print(f"[GraniteDocling] Found {len(tables)} tables")

        total_time = time.time() - start_time

        return {
            "status": "success",
            "result": {
                "doctags": doctags,
                "html": html_output,
                "tables": tables,
                "metadata": {
                    "model": "granite-docling-258M",
                    "inference_time_seconds": round(inference_time, 2),
                    "total_time_seconds": round(total_time, 2),
                    "table_count": len(tables)
                }
            }
        }

    except Exception as e:
        print(f"[GraniteDocling] Error: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    print("[GraniteDocling] Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
