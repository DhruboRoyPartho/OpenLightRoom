import numpy as np

def adjust_blacks(image: np.ndarray, value: float) -> np.ndarray:
    # Normalize image
    image = image.astype(np.float32) / 255.0

    threshold = 0.2
    blacks = (threshold - image) / threshold
    blacks = np.clip(blacks, 0.0, 1.0)

    blacks *= value / 100.0
    image += blacks

    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)