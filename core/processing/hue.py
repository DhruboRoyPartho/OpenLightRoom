import cv2
import numpy as np

def adjust_hue(image: np.ndarray, value: float) -> np.ndarray:
    # value range: -180..180 degrees. A global rotation of every pixel's
    # hue around the color wheel, leaving saturation and lightness alone.
    if value == 0:
        return image

    clipped = np.clip(image, 0.0, 1.0).astype(np.float32)
    hsv = cv2.cvtColor(clipped, cv2.COLOR_RGB2HSV)

    hsv[:, :, 0] = (hsv[:, :, 0] + value) % 360.0

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
