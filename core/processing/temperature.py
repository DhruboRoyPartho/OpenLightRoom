import numpy as np

def adjust_temperature(image: np.ndarray, value: float) -> np.ndarray:
    image = image.astype(np.float32)

    # Normalize temp value
    temp_scale = 1+value/100.0

    # Blue-Yellow Channel color shift
    image[:,:,0] *= temp_scale
    image[:,:,2] /= temp_scale

    image = np.clip(image, 0, 255).astype(np.uint8)
    return image