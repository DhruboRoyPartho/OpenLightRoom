import numpy as np

def adjust_brightness(image: np.ndarray, value: float) -> np.ndarray:
    # value range: -1.0 to +1.0
    image = image.astype(np.float32) / 255.0
    image += value
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)
