from cog import BasePredictor, Input, Path
from PIL import Image
from transformers import AutoModelForCausalLM
import tempfile
import torch

WEIGHTS_DIR = "/src/weights"


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            WEIGHTS_DIR,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.cuda()
        self.model.eval()
        self.processor = self.model.model.processor

    def predict(
        self,
        image: Path = Input(description="Input image (PNG/JPG) to convert to SVG."),
        max_length: int = Input(
            description="Maximum number of tokens to generate for the SVG.",
            default=4000,
            ge=256,
            le=8000,
        ),
    ) -> Path:
        image_pil = Image.open(str(image)).convert("RGB")
        pixel_values = self.processor(image_pil, return_tensors="pt")["pixel_values"].cuda()
        if pixel_values.shape[0] != 1:
            pixel_values = pixel_values.squeeze(0)
        batch = {"image": pixel_values}

        with torch.no_grad():
            raw_svg = self.model.generate_im2svg(batch, max_length=max_length)[0]

        try:
            from starvector.data.util import process_and_rasterize_svg
            svg, _ = process_and_rasterize_svg(raw_svg)
        except Exception:
            svg = raw_svg

        out_path = Path(tempfile.mkstemp(suffix=".svg")[1])
        out_path.write_text(svg)
        return out_path
