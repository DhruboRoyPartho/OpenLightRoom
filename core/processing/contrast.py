import numpy as np

def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    # Factor = 1.0 means no change, <1 reduces contrast, >1 increases
    image = image.astype(np.float32) / 255
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * factor + mean
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)