import numpy as np

def adjust_exposure(image: np.ndarray, value: float) -> np.ndarray:
    image = image.astype(np.float32)

    value /= 20.0   # Mapping -5.0 to 5.0

    image = image * (2 ** value)

    image = np.clip(image, 0, 255).astype(np.uint8)
    return image