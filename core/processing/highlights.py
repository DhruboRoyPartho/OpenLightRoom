import numpy as np
from core.processing.color_space import luminance, smoothstep

def adjust_highlights(image: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return image

    luma = luminance(image)

    # A smooth weight that engages gradually through the upper tonal range,
    # computed once from luminance and applied equally to all channels, so
    # highlight recovery doesn't produce a hard-edged band (banding/halos)
    # or shift hue the way independent per-channel thresholds would.
    lo, hi = 0.35, 1.0
    t = np.clip((luma - lo) / (hi - lo), 0.0, 1.0)
    weight = smoothstep(t)

    return image + (weight * (value / 100.0))[..., None]
