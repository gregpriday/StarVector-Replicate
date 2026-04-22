"""Local CPU smoke test for the StarVector deployment code.

Uses starvector-1b-im2svg (smaller variant, ~2GB) to validate the end-to-end
inference path on CPU before pushing the 8B model to Replicate. If this runs,
the custom trust_remote_code path, processor access pattern, and
generate_im2svg call in predict.py are validated.
"""
import os

os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

import sys
import time

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM

MODEL_ID = os.environ.get("SMOKE_MODEL", "starvector/starvector-8b-im2svg")
MAX_LENGTH = int(os.environ.get("SMOKE_MAX_LENGTH", "650"))


def make_test_image(path="test_input.png"):
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((50, 50, 200, 200), fill="red", outline="black", width=4)
    draw.rectangle((90, 90, 160, 160), fill="yellow", outline="black", width=3)
    img.save(path)
    return path


def main():
    print(f"[1/5] Creating test image")
    img_path = make_test_image()

    print(f"[2/5] Loading {MODEL_ID} on CPU (bfloat16, eager attention) — downloads on first run")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"      loaded in {time.time() - t0:.1f}s")

    print(f"[3/5] Accessing processor via model.model.processor")
    processor = model.model.processor

    print(f"[4/5] Running processor on test image")
    image_pil = Image.open(img_path).convert("RGB")
    pixel_values = processor(image_pil, return_tensors="pt")["pixel_values"]
    if pixel_values.shape[0] != 1:
        pixel_values = pixel_values.squeeze(0)
    batch = {"image": pixel_values}
    print(f"      pixel_values shape: {tuple(pixel_values.shape)}")

    print(f"[5/5] Generating SVG (max_length={MAX_LENGTH}) — CPU, may take minutes")
    t0 = time.time()
    with torch.no_grad():
        raw_svg = model.generate_im2svg(
            batch,
            max_length=MAX_LENGTH,
            num_beams=1,
            use_nucleus_sampling=False,
        )[0]
    print(f"      generated in {time.time() - t0:.1f}s")

    print("\n--- RAW SVG (first 500 chars) ---")
    print(raw_svg[:500])
    print("--- END ---\n")

    with open("test_output_raw.svg", "w") as f:
        f.write(raw_svg)
    print("Wrote test_output_raw.svg")

    try:
        from starvector.data.util import process_and_rasterize_svg
        svg, _ = process_and_rasterize_svg(raw_svg)
        with open("test_output_clean.svg", "w") as f:
            f.write(svg)
        print("Wrote test_output_clean.svg (post-processed)")
    except Exception as e:
        print(f"(skipped post-processing: {type(e).__name__}: {e})")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
