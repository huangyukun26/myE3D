from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .base import NormalPredictor


class MockPredictor(NormalPredictor):
    """Dependency-free normal predictor.

    By default, returns a constant "facing camera" normal (0.5, 0.5, 1.0)
    encoded to 8-bit RGB as [128, 128, 255]. If a coarse normal provider is
    available, it will prefer that output.
    """

    def __init__(self, coarse_normal_provider: Optional[Callable[[], Optional[Image.Image]]] = None):
        self.coarse_normal_provider = coarse_normal_provider

    def __call__(self, pil_rgb: Image.Image) -> Tuple[Image.Image, Dict]:
        coarse_normal = None
        if self.coarse_normal_provider is not None:
            coarse_normal = self.coarse_normal_provider()

        if coarse_normal is not None:
            normal_rgb = coarse_normal.convert("RGB")
            return normal_rgb, {"source": "coarse_normal"}

        normal_arr = np.zeros((pil_rgb.height, pil_rgb.width, 3), dtype=np.uint8)
        normal_arr[..., 0] = 128
        normal_arr[..., 1] = 128
        normal_arr[..., 2] = 255
        normal_img = Image.fromarray(normal_arr, mode="RGB")
        return normal_img, {"source": "mock_constant"}
