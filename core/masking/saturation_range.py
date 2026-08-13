import numpy as np

from core.masking.range_mask import range_mask


def saturation_range_mask(saturation: np.ndarray, low: float, high: float, feather: float) -> np.ndarray:
    """1.0 for saturation inside [low, high] (saturation expected in
    [0, 1]), smoothly feathered on both sides. Thin wrapper over
    range_mask for readability at call sites."""
    return range_mask(saturation, low, high, feather)
