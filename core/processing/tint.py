import numpy as np

def adjust_tint(image: np.ndarray, value: float) -> np.ndarray:
    # value range: -100 (green) .. 100 (magenta). Same linear-gain
    # reasoning as temperature, on the green channel.
    if value == 0:
        return image
    factor = 2.0 ** (-value / 100.0)  # positive value (magenta) reduces green
    result = image.copy()
    result[:, :, 1] *= factor
    return result
