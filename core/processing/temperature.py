import numpy as np

def adjust_temperature(image: np.ndarray, value: float) -> np.ndarray:
    # value range: -100 (cooler/blue) .. 100 (warmer/yellow). White balance
    # is physically a per-channel gain applied to linear light - exactly
    # what a camera's WB coefficients do - so a straightforward channel
    # multiply is correct now that this runs in the linear working stage
    # (unlike scaling gamma-encoded bytes, which distorts luminance).
    if value == 0:
        return image
    factor = 2.0 ** (value / 100.0)  # always positive - no div-by-zero at the extremes
    result = image.copy()
    result[:, :, 0] *= factor   # red
    result[:, :, 2] /= factor   # blue
    return result
