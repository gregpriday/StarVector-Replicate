# StarVector on Replicate

A Replicate deployment wrapper for [StarVector 8B](https://huggingface.co/starvector/starvector-8b-im2svg), the image-to-SVG model from Rodriguez et al. at ServiceNow Research (CVPR 2024).

This repo isn't the model itself. It's the thin layer that packages StarVector into a [Cog](https://cog.run) container and auto-deploys it to Replicate via GitHub Actions. All the clever work (the architecture, the training, the weights) belongs to the original authors.

## Live model

https://replicate.com/gregpriday/starvector-8b-im2svg

Give it a raster image (PNG/JPG) of a logo, icon, or simple illustration, and it produces SVG code that reconstructs the image as vector paths. Works best on content it was trained on: logos, icons, diagrams, simple graphics. It struggles with photos, dense textures, or anything overly abstract.

## Usage

Via the Replicate HTTP API:

```bash
curl -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "2cd3242be97ee8aeb815993501328ff7dad162ffe9a5b33ef8fbf4fb66410669",
    "input": {
      "image": "https://example.com/your-logo.png",
      "max_length": 4000,
      "num_beams": 2
    }
  }'
```

Or with the Python client:

```python
import replicate

output = replicate.run(
    "gregpriday/starvector-8b-im2svg:2cd3242be97ee8aeb815993501328ff7dad162ffe9a5b33ef8fbf4fb66410669",
    input={
        "image": open("logo.png", "rb"),
        "max_length": 4000,
        "num_beams": 2,
    },
)
# output is a URL to the generated SVG file
```

### Inputs

| Input | Default | Notes |
|-------|---------|-------|
| `image` | required | The raster image to vectorise. |
| `max_length` | 4000 | Total token length (includes ~578 prompt tokens for the image + `<svg` prefix). |
| `num_beams` | 2 | Beam search width. `1` is greedy (fastest). Higher values trade compute for cleaner output. |
| `temperature` | 1.0 | Sampling temperature. Only used when `num_beams=1`. |

On an L40S, expect roughly 20 to 90 seconds of inference per request, depending on `max_length` and `num_beams`. Cold boots currently take a few minutes because the model loads StarCoder2-7B weights on first setup.

## How the deploy works

There's no local Docker involved. Every push to `main` triggers a GitHub Actions workflow that:

1. Reclaims about 30 GB of disk on the runner (jlumbroso/free-disk-space).
2. Installs Cog via the official [replicate/setup-cog](https://github.com/replicate/setup-cog) action.
3. Runs `cog push r8.im/gregpriday/starvector-8b-im2svg`, which builds the CUDA image and pushes it to Replicate's registry. Replicate then creates a new model version.

The build takes roughly 17 minutes end to end. The StarVector weights (~16 GB) are baked into the image during the build step so cold boots don't re-download them. StarCoder2-7B is still pulled at first setup, which accounts for most of the cold-start time. Baking it in as well would push the image past the free runner's disk budget.

Authentication is a single repository secret, `REPLICATE_CLI_AUTH_TOKEN`.

## Local smoke test

`smoke.py` runs the full inference pipeline on CPU against the 8B model. It's useful for validating dependency changes without waiting for a GitHub Actions rebuild.

```bash
uv venv --python 3.11 .venv_smoke
VIRTUAL_ENV=$(pwd)/.venv_smoke uv pip install -r requirements-smoke.txt
./.venv_smoke/bin/python smoke.py
```

Note: StarVector's modeling code hardcodes `attn_implementation="flash_attention_2"` in `starcoder2.py`, which won't load on CPU. The smoke test patches that to `eager` in the installed venv. The Replicate build path is unaffected because flash-attn 2 installs cleanly on CUDA.

Running on CPU is slow (about 1 tok/sec on Apple Silicon) and needs roughly 32 GB of free RAM.

## Credits

**StarVector** is the work of:

Juan A. Rodriguez, Abhay Puri, Shubham Agarwal, Issam H. Laradji, Pau Rodriguez, Sai Rajeswar, David Vazquez, Christopher Pal, and Marco Pedersoli.

> Rodriguez, J. A., et al. *StarVector: Generating Scalable Vector Graphics Code from Images and Text*. CVPR 2024.

- Paper: https://arxiv.org/abs/2312.11556
- Project page: https://starvector.github.io
- Original code: https://github.com/joanrod/star-vector
- Model weights: https://huggingface.co/starvector/starvector-8b-im2svg

The 8B variant builds on two other open models:

- [StarCoder2-7B](https://huggingface.co/bigcode/starcoder2-7b) (BigCode) as the code generation backbone.
- [SigLIP-384](https://huggingface.co/google/siglip-so400m-patch14-384) (Google) as the image encoder.

## Licence

The wrapper code in this repo (cog.yaml, predict.py, smoke.py, the GitHub Actions workflow) is Apache 2.0.

The StarVector weights and original model code are Apache 2.0 per [their pyproject.toml](https://github.com/joanrod/star-vector/blob/main/pyproject.toml). The StarCoder2 base model ships under [BigCode OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement), which has its own use restrictions you should read before deploying anything downstream.
