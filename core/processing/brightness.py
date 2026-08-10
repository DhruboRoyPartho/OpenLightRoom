import numpy as np

def adjust_brightness(image: np.ndarray, value: float) -> np.ndarray:
    # value range: -1.0 to +1.0. Operates on display-referred float data.
    if value == 0:
        return image

    # Weight the shift by distance from the midtones so brightness eases off
    # near black/white instead of clipping hard at the extremes - keeps
    # highlight and shadow detail intact, unlike a flat additive offset.
    weight = 1.0 - np.abs(np.clip(image, 0.0, 1.0) * 2.0 - 1.0)
    return image + value * weight
