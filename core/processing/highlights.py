import numpy as np

def adjust_highlights(image: np.ndarray, value: float) -> np.ndarray:
    # Normalize
    image = image.astype(np.float32) / 255.0
    mask = image > 0.6  # mask for brighter area
    image[mask] += value / 100.0
    image = np.clip(image, 0.0, 1.0)

    return (image * 255).astype(np.uint8)