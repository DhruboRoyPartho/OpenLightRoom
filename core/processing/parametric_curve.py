import numpy as np
from core.processing.color_space import luminance, smoothstep

# Fixed region centers over the 0-255 tone scale, matching a standard
# quartered parametric curve (Shadows / Darks / Lights / Highlights).
REGION_CENTERS_255 = (0.0, 85.0, 170.0, 255.0)
MAX_SHIFT = 0.5  # how far a +/-100 region amount can push that region

# Property used below: for the standard cubic smoothstep,
# smoothstep(t) + smoothstep(1 - t) == 1 for all t. That makes each pair of
# neighboring region weights sum to exactly 1 between their centers, so the
# four regions blend smoothly with no gaps or double-counting.


def _region_weights(x: np.ndarray):
    """x: float array of tone values in [0, 1]. Returns (w_shadows,
    w_darks, w_lights, w_highlights), each the same shape as x, summing to
    1.0 at every point - a weight is 1 exactly at its own region's center
    and 0 at each neighboring center, smoothly interpolated between."""
    centers = [c / 255.0 for c in REGION_CENTERS_255]
    weights = []
    for i, center in enumerate(centers):
        left = centers[i - 1] if i > 0 else None
        right = centers[i + 1] if i < len(centers) - 1 else None

        w = np.ones_like(x, dtype=np.float64)
        if left is not None:
            t = np.clip((x - left) / (center - left), 0.0, 1.0)
            w = np.where(x < center, smoothstep(t), w)
        if right is not None:
            t = np.clip((right - x) / (right - center), 0.0, 1.0)
            w = np.where(x > center, smoothstep(t), w)
        weights.append(w)
    return weights


def apply_parametric_curve(image: np.ndarray, highlights: float, lights: float,
                            darks: float, shadows: float) -> np.ndarray:
    """Each amount is -100..100. Shapes the tone curve by region instead of
    dragging individual points - the region an amount affects is
    determined by luminance, with smooth overlap into neighboring regions
    so there's no visible seam, the same technique Highlights/Shadows/
    Whites/Blacks already use."""
    if highlights == 0 and lights == 0 and darks == 0 and shadows == 0:
        return image

    luma = np.clip(luminance(image), 0.0, 1.0)
    w_shadows, w_darks, w_lights, w_highlights = _region_weights(luma)

    shift = (
        (shadows / 100.0) * w_shadows
        + (darks / 100.0) * w_darks
        + (lights / 100.0) * w_lights
        + (highlights / 100.0) * w_highlights
    ) * MAX_SHIFT

    return image + shift[..., None]
