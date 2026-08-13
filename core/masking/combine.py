import numpy as np


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """Intersect multiple [0, 1] masks (e.g. hue AND saturation AND
    luminance range) by elementwise product - each input mask independently
    ranges 0..1, so multiplying keeps the result in 0..1 and requires every
    condition to hold at full strength to reach 1.0, without needing a hard
    boolean AND (which would reintroduce the hard edges this module exists
    to avoid)."""
    if not masks:
        raise ValueError("combine_masks() requires at least one mask")
    result = masks[0]
    for m in masks[1:]:
        result = result * m
    return np.clip(result, 0.0, 1.0)
