import numpy as np
from core.processing.color_space import luminance, smoothstep

def adjust_shadows(image: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return image

    luma = luminance(image)

    # Mirror of highlights: a smooth weight over the lower tonal range,
    # shared across channels so shadow recovery doesn't shift hue.
    lo, hi = 0.0, 0.45
    t = np.clip((hi - luma) / (hi - lo), 0.0, 1.0)
    weight = smoothstep(t)

    return image + (weight * (value / 100.0))[..., None]
