from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

from core.masking.hue_range import hue_range_mask
from core.masking.saturation_range import saturation_range_mask
from core.masking.luminance_range import luminance_zone_mask
from core.masking.combine import combine_masks
from core.processing.color_space import luminance


@dataclass(frozen=True)
class SelectiveColorMask:
    """A reusable, combinable color-region selection: an optional hue
    range, an optional saturation range, and an optional luminance range,
    intersected together into a single [0, 1] mask. Any axis can be left
    unset (None) to select without restriction on that axis - e.g. one HSL
    channel (Red, Orange, ...) is a SelectiveColorMask with only the hue
    fields set; a shadows/highlights split is one with only the luminance
    fields set. This is the one place "which pixels does this edit affect"
    is computed, so HSL, Color Wheels, and Selective Color all build on the
    same smooth, feathered selection logic instead of each re-deriving
    their own hard-edged threshold.

    Saturation and value/lightness are read from HSV (consistent with the
    existing Hue/Saturation/Vibrance tools); luminance uses the same
    Rec. 709 luma as the existing Highlights/Shadows/Whites/Blacks tools,
    so a luminance-range selection lines up with what those tools already
    consider "shadows" or "highlights".
    """

    hue_center_deg: Optional[float] = None
    hue_width_deg: float = 30.0
    hue_feather_deg: float = 30.0

    sat_low: Optional[float] = None
    sat_high: float = 1.0
    sat_feather: float = 0.1

    luma_low: Optional[float] = None
    luma_high: float = 1.0
    luma_feather: float = 0.2

    def evaluate(self, image: np.ndarray) -> np.ndarray:
        """image: float32 RGB, HxWx3, values expected in [0, 1] (any
        encoding - display or linear - since hue/saturation/luminance are
        all defined relative to the image as given, not to a particular
        color space). Returns a float32 HxW mask in [0, 1].
        """
        clipped = np.clip(image, 0.0, 1.0).astype(np.float32)

        masks = []
        if self.hue_center_deg is not None or self.sat_low is not None:
            hsv = cv2.cvtColor(clipped, cv2.COLOR_RGB2HSV)
            if self.hue_center_deg is not None:
                masks.append(hue_range_mask(hsv[..., 0], self.hue_center_deg, self.hue_width_deg, self.hue_feather_deg))
            if self.sat_low is not None:
                masks.append(saturation_range_mask(hsv[..., 1], self.sat_low, self.sat_high, self.sat_feather))
        if self.luma_low is not None:
            masks.append(luminance_zone_mask(luminance(clipped), self.luma_low, self.luma_high, self.luma_feather))

        if not masks:
            return np.ones(image.shape[:2], dtype=np.float32)
        return combine_masks(*masks).astype(np.float32)
