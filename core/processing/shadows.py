import numpy as np

def adjust_shadows(image: np.ndarray, value: float) -> np.ndarray:
    # Normalize image
    image = image.astype(np.float32) / 255.0

    mask = image < 0.35

    image[mask] += value / 100.0

    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)