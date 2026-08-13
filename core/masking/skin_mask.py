import numpy as np

from core.scopes.vectorscope import rgb_to_cb_cr
from core.masking.range_mask import range_mask

# The classic YCbCr skin-tone band (after Chai & Ngan and the many
# variants descended from it): real, deterministic computer vision, not a
# claim of AI/ML - skin tones across a wide range of ethnicities and
# lighting cluster in this Cb/Cr band regardless of luma. This is the same
# YCbCr color-difference plane the app's Vectorscope already uses for its
# skin-tone reference line (core/scopes/vectorscope.py:SKIN_TONE_ANGLE_DEG),
# reused here rather than a second color-difference implementation.
_CB_LOW, _CB_HIGH = 77.0 / 255.0 - 0.5, 127.0 / 255.0 - 0.5
_CR_LOW, _CR_HIGH = 133.0 / 255.0 - 0.5, 173.0 / 255.0 - 0.5


def skin_mask(image: np.ndarray, feather: float = 20.0) -> np.ndarray:
    """image: float32 RGB, HxWx3, in [0, 1]. A heuristic, not machine
    learning: it will pick up other orange/tan-colored content too (wood,
    some fabrics, warm-toned skies), which is exactly why it's meant to be
    combined with the app's other mask types (Intersect with a Subject or
    a Brush mask, Subtract a Color Range for the offending fabric) rather
    than trusted alone.
    """
    cb, cr = rgb_to_cb_cr(image)
    feather_span = max(feather, 1.0) / 100.0 * 0.2
    cb_mask = range_mask(cb, _CB_LOW, _CB_HIGH, feather_span)
    cr_mask = range_mask(cr, _CR_LOW, _CR_HIGH, feather_span)
    return np.clip(cb_mask * cr_mask, 0.0, 1.0).astype(np.float32)
