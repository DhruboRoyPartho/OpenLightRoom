import numpy as np

from core.masking.range_mask import range_mask
from core.processing.color_space import smoothstep


def luminance_zone_mask(luma: np.ndarray, low: float, high: float, feather: float) -> np.ndarray:
    """1.0 for luma inside [low, high] (luma expected in [0, 1]), smoothly
    feathered on both sides. Thin wrapper over range_mask for readability
    at call sites."""
    return range_mask(luma, low, high, feather)


def three_zone_luminance_masks(luma: np.ndarray, shadow_edge: float = 0.33,
                                highlight_edge: float = 0.66, feather: float = 0.2):
    """Shadows/Midtones/Highlights tonal-separation masks that form a
    partition of unity - shadows + midtones + highlights == 1.0 at every
    pixel - by construction, exploiting the identity
    smoothstep(t) + smoothstep(1-t) == 1. This is what makes a tool built
    on these masks (color wheels, split toning) blend seamlessly across
    zone boundaries instead of showing a hard seam or double-counting a
    pixel in two zones.

    `feather` is the ramp's full width (the ramp runs from edge-feather to
    edge+feather); shadow_edge must be < highlight_edge, and `feather`
    should be small relative to (highlight_edge - shadow_edge) or the two
    ramps overlap and the exact partition-of-unity property degrades
    (silently clamped rather than raising, since a live UI slider can pass
    transient out-of-order values while being dragged).

    Returns (shadows, midtones, highlights), each the same shape as luma.
    """
    feather = max(feather, 1e-6)

    shadow_to_mid = smoothstep(np.clip((luma - (shadow_edge - feather)) / (2.0 * feather), 0.0, 1.0))
    mid_to_highlight = smoothstep(np.clip((luma - (highlight_edge - feather)) / (2.0 * feather), 0.0, 1.0))

    shadows = 1.0 - shadow_to_mid
    highlights = mid_to_highlight
    midtones = np.clip(shadow_to_mid - mid_to_highlight, 0.0, 1.0)

    return shadows, midtones, highlights
