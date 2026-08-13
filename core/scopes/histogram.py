import numpy as np

from core.scopes._common import clean_image
from core.processing.color_space import luminance

CHANNELS = ("R", "G", "B")


def compute_rgb_histogram(image: np.ndarray, bins: int = 256) -> dict:
    """image: float32 RGB, HxWx3, in [0, 1] (display-referred). Returns
    {"R": counts, "G": counts, "B": counts}, each an int64 array of length
    `bins` - per-bin pixel counts across [0, 1], via np.histogram (a single
    vectorized C call per channel, not a Python per-pixel loop)."""
    clipped = clean_image(image)
    return {
        ch: np.histogram(clipped[..., i], bins=bins, range=(0.0, 1.0))[0]
        for i, ch in enumerate(CHANNELS)
    }


def compute_luminance_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Same as compute_rgb_histogram, but of Rec. 709 luma (the same
    definition used by the app's tonal tools - see
    core/processing/color_space.py) rather than one raw channel."""
    clipped = clean_image(image)
    luma = luminance(clipped)
    counts, _ = np.histogram(luma, bins=bins, range=(0.0, 1.0))
    return counts
