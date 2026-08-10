import cv2
import numpy as np

def adjust_vibrance(image: np.ndarray, value: float) -> np.ndarray:
    # value range: -100..100. Unlike a flat Saturation scale, the boost is
    # weighted by how saturated a pixel already is: raising vibrance lifts
    # muted colors (skin tones, pale skies) more than already-vivid ones,
    # so it protects against oversaturating what's already saturated.
    # Lowering vibrance is the mirror image - it pulls down the most
    # saturated colors first, leaving muted ones alone. This is the
    # standard vibrance formulation used across most photo editors.
    if value == 0:
        return image

    clipped = np.clip(image, 0.0, 1.0).astype(np.float32)
    hsv = cv2.cvtColor(clipped, cv2.COLOR_RGB2HSV)

    s = hsv[:, :, 1]
    weight = (1.0 - s) if value > 0 else s
    factor = 1.0 + (value / 100.0) * weight
    hsv[:, :, 1] = np.clip(s * factor, 0.0, 1.0)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
