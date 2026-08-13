import numpy as np

from core.processing.color_space import smoothstep


def range_mask(value: np.ndarray, low: float, high: float, feather: float) -> np.ndarray:
    """Smooth mask: 1.0 for `value` inside [low, high], easing to 0.0 over
    an additional `feather` on each side via smoothstep - a feathered
    selection instead of a hard boolean threshold, so a tool built on this
    (tonal separation, saturation range, etc.) doesn't produce banding at
    the selection edge. Not circular - for bounded, non-wrapping ranges
    like saturation or luminance (see hue_range.py for the wrap-around
    case). `feather` must be > 0; degenerately small values approximate a
    hard edge without dividing by zero.
    """
    feather = max(feather, 1e-6)
    rising = smoothstep(np.clip((value - (low - feather)) / feather, 0.0, 1.0))
    falling = smoothstep(np.clip(((high + feather) - value) / feather, 0.0, 1.0))
    return np.clip(rising * falling, 0.0, 1.0)
