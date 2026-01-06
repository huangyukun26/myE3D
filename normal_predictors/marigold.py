from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

from PIL import Image

from .base import NormalPredictor


class MarigoldPredictor(NormalPredictor):
    """Marigold-based normal predictor wrapper."""

    def __init__(self, cfg, device: str):
        from Mono.Marigold import setup_mari_pipeline

        mari_pipe = setup_mari_pipeline(cfg.mono)
        mari_pipe = mari_pipe.to(device)
        mari_pipe.unet.eval()

        self._mari_pipe = partial(
            mari_pipe,
            denoising_steps=cfg.mono.denoise_steps,
            ensemble_size=cfg.mono.ensemble_size,
            processing_res=cfg.mono.processing_res,
            match_input_res=True,
            batch_size=cfg.mono.batch_size,
            color_map=cfg.mono.color_map,
            show_progress_bar=False,
            resample_method=cfg.mono.resample_method,
            normals=(cfg.mono.modality == "normals"),
            noise=cfg.mono.noise,
        )

    def __call__(self, pil_rgb: Image.Image) -> Tuple[Image.Image, Dict]:
        mari_pipe_out = self._mari_pipe(pil_rgb)
        normal_img = mari_pipe_out.normal_colored
        return normal_img, {"mari_output": mari_pipe_out}
