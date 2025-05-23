import numpy as np

def adjust_tint(image: np.ndarray, value: float) -> np.ndarray:
    image = image.astype(np.float32)
    value = value * (-1)

    # Normalize temp value
    tint_scale = 1+value/100.0

    # Green Magenta shift in green channel
    image[:,:,1] *= tint_scale

    image = np.clip(image, 0, 255).astype(np.uint8)
    return image