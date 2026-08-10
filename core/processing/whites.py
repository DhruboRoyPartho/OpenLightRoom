import numpy as np
from core.processing.color_space import luminance, smoothstep

def adjust_whites(image: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return image

    luma = luminance(image)

    # Sets the white clipping point: a smooth, luminance-based weight (not
    # independent per-channel thresholds) so the near-white region shifts
    # together without introducing a color cast.
    threshold = 0.8
    t = np.clip((luma - threshold) / (1.0 - threshold), 0.0, 1.0)
    weight = smoothstep(t)

    return image + (weight * (value / 100.0))[..., None]
