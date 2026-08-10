import numpy as np

# Standard sRGB <-> linear-light conversion (IEC 61966-2-1). Adjustments that
# represent a physical quantity - a stop of exposure is a doubling of linear
# light, not a doubling of the gamma-encoded byte value - need to convert
# into linear light first or they look wrong (highlights clip too early,
# stops don't feel equal-sized).

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """img: float32 in [0, 1], gamma-encoded (sRGB). Returns linear-light."""
    a = 0.055
    return np.where(img <= 0.04045, img / 12.92, ((img + a) / (1 + a)) ** 2.4)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """img: float32 in [0, 1], linear-light. Returns gamma-encoded (sRGB)."""
    a = 0.055
    img = np.clip(img, 0.0, None)
    return np.where(img <= 0.0031308, img * 12.92, (1 + a) * np.power(img, 1 / 2.4) - a)


def linear_to_display(img: np.ndarray) -> np.ndarray:
    """Scene-linear working image -> display-referred (sRGB gamma encoded).
    This is the single point in the render pipeline where linear-light data
    (from a RAW demosaic, or a decoded JPEG/PNG) becomes "what you see":
    White Balance and Exposure operate before this call, in linear light,
    which is where those operations are physically meaningful; every other
    tone tool (Brightness, Contrast, Highlights/Shadows/Whites/Blacks,
    Curve) operates after it, on the perceptually-encoded result, which is
    where those *parametric* tone tools were designed and tuned to behave
    intuitively. A generous headroom clip is applied first so recoverable
    highlight detail well above nominal white doesn't produce runaway
    values once gamma-encoded.
    """
    return linear_to_srgb(np.clip(img, 0.0, 16.0))


def luminance(img: np.ndarray) -> np.ndarray:
    """Perceptual luma (Rec. 709 weights) of an HxWx3 RGB float image."""
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def smoothstep(t: np.ndarray) -> np.ndarray:
    """Smooth 0->1 easing for tonal-range masks, avoiding the hard edges
    (and resulting banding/halos) of a plain threshold."""
    return t * t * (3.0 - 2.0 * t)
