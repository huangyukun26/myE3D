from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from PIL import Image


class NormalPredictor(ABC):
    """Abstract normal predictor interface.

    Implementations must return an RGB PIL.Image normal map in 0-255 encoding,
    compatible with downstream load_and_blend_maps.
    """

    @abstractmethod
    def __call__(self, pil_rgb: Image.Image) -> Tuple[Image.Image, Dict]:
        """Predict a normal map from an RGB input.

        Args:
            pil_rgb: RGB input image.

        Returns:
            (normal_pil, extra_dict)
        """
        raise NotImplementedError
