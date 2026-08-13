import cv2
import numpy as np

from core.masking.selective_color_mask import SelectiveColorMask


def color_range_mask(image: np.ndarray, sample_rgb, refine: float = 50.0) -> np.ndarray:
    """image: float32 RGB, HxWx3, in [0, 1]. sample_rgb: (r, g, b) in
    [0, 1] - the eyedropper-picked reference color. refine: 0..100, how
    tightly the selection hugs that exact color; 0 is a loose, broad
    selection of similar colors, 100 is a tight, near-exact match only -
    Lightroom's Color Range mask's "Refine" slider.

    Built on the same smooth hue/saturation-range machinery the app's HSL
    and Selective Color tools already use (see
    core/masking/selective_color_mask.py) rather than a separate
    implementation, so a Color Range mask feathers the same way those do.
    """
    r, g, b = sample_rgb
    sample = np.array([[[r, g, b]]], dtype=np.float32)
    hsv = cv2.cvtColor(np.clip(sample, 0.0, 1.0), cv2.COLOR_RGB2HSV)
    hue, sat, _val = hsv[0, 0]

    t = np.clip(refine, 0.0, 100.0) / 100.0
    hue_width = 60.0 * (1.0 - t) + 5.0     # loose (t=0): 65 deg total; tight (t=1): 5 deg
    hue_feather = hue_width
    sat_span = 0.6 * (1.0 - t) + 0.05

    mask = SelectiveColorMask(
        hue_center_deg=float(hue), hue_width_deg=hue_width, hue_feather_deg=hue_feather,
        sat_low=max(0.0, float(sat) - sat_span), sat_high=min(1.0, float(sat) + sat_span), sat_feather=0.15,
    )
    return mask.evaluate(image)
