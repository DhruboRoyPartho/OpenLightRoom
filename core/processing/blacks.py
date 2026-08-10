import numpy as np
from core.processing.color_space import luminance, smoothstep

def adjust_blacks(image: np.ndarray, value: float) -> np.ndarray:
    if value == 0:
        return image

    luma = luminance(image)

    # Sets the black clipping point, mirroring whites: a smooth,
    # luminance-based weight so the near-black region shifts together
    # without introducing a color cast.
    threshold = 0.2
    t = np.clip((threshold - luma) / threshold, 0.0, 1.0)
    weight = smoothstep(t)

    return image + (weight * (value / 100.0))[..., None]
