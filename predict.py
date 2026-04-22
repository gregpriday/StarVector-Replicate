import tempfile

import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from transformers import AutoModelForCausalLM

WEIGHTS_DIR = "/src/weights"


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            WEIGHTS_DIR,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.cuda()
        self.model.eval()
        self.processor = self.model.model.processor

    def predict(
        self,
        image: Path = Input(description="Input image (PNG/JPG) to convert to SVG."),
        max_length: int = Input(
            description="Maximum total token length for the generated SVG (includes ~578 prompt tokens).",
            default=4000,
            ge=600,
            le=16000,
        ),
        num_beams: int = Input(
            description="Beam search width. 1 = greedy (fastest). Higher values may produce cleaner SVGs at extra cost.",
            default=2,
            ge=1,
            le=4,
        ),
        temperature: float = Input(
            description="Sampling temperature. Only used when num_beams=1.",
            default=1.0,
            ge=0.1,
            le=2.0,
        ),
    ) -> Path:
        image_pil = Image.open(str(image)).convert("RGB")
        pixel_values = self.processor(image_pil, return_tensors="pt")["pixel_values"].cuda()
        if pixel_values.shape[0] != 1:
            pixel_values = pixel_values.squeeze(0)
        batch = {"image": pixel_values}

        with torch.no_grad():
            raw_svg = self.model.generate_im2svg(
                batch,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                use_nucleus_sampling=(num_beams == 1),
            )[0]

        try:
            from starvector.data.util import process_and_rasterize_svg
            svg, _ = process_and_rasterize_svg(raw_svg)
        except Exception:
            svg = raw_svg

        fd, tmp_path = tempfile.mkstemp(suffix=".svg")
        with open(fd, "w") as f:
            f.write(svg)
        return Path(tmp_path)
