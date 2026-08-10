import cv2
import numpy as np

def adjust_saturation(image: np.ndarray, value: float) -> np.ndarray:
    # value range: -100 (grayscale) .. 100 (double saturation). A uniform
    # scale of every pixel's saturation equally, unlike Vibrance which
    # protects already-saturated colors.
    if value == 0:
        return image

    clipped = np.clip(image, 0.0, 1.0).astype(np.float32)
    hsv = cv2.cvtColor(clipped, cv2.COLOR_RGB2HSV)

    factor = 1.0 + value / 100.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0.0, 1.0)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
