import numpy as np

def adjust_whites(image: np.ndarray, value: float) -> np.ndarray:
    # Normalize image
    image = image.astype(np.float32) / 255.0

    threshold = 0.8

    whites = (image - threshold) / (1.0 - threshold)
    whites = np.clip(whites, 0.0, 1.0)

    whites *= value / 100.0

    image += whites

    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)