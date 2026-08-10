import numpy as np

def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    # Factor = 1.0 means no change, <1 reduces contrast, >1 increases.
    # Operates on display-referred float data.
    if factor == 1.0:
        return image

    # Pivot around a fixed middle gray rather than the image's own mean, so
    # the same slider value always produces the same result regardless of
    # what other edits already changed the image's brightness distribution.
    pivot = 0.5
    return (image - pivot) * factor + pivot
