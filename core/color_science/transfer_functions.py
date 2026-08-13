"""Electro-optical transfer functions (gamma/OETF/EOTF curves).

Each pair converts between a display-referred (gamma-encoded) signal in
[0, 1] and scene-linear light. These are pure, UI-independent math - no
image-processing policy lives here, just the transfer curves themselves as
published by their respective standards. All functions are vectorized
(numpy ufuncs) and safe for arrays of any shape.

Kept separate from the interpolation/masking/blending logic in
core/processing/ so a single, tested definition of "what does sRGF gamma
actually do" is shared by the render pipeline, the color-space conversion
layer, and any future RAW/ICC work - see core/color_science/spaces.py for
how these compose with primaries into full ColorSpace definitions.
"""

import numpy as np


def srgb_eotf(encoded: np.ndarray) -> np.ndarray:
    """Display-referred sRGB [0,1] -> linear light. IEC 61966-2-1."""
    a = 0.055
    x = np.asarray(encoded, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def srgb_oetf(linear: np.ndarray) -> np.ndarray:
    """Linear light -> display-referred sRGB [0,1]. IEC 61966-2-1."""
    a = 0.055
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * np.power(x, 1 / 2.4) - a)


def rec709_eotf(encoded: np.ndarray) -> np.ndarray:
    """Display-referred Rec.709/BT.709 [0,1] -> linear light."""
    x = np.asarray(encoded, dtype=np.float64)
    return np.where(x < 0.081, x / 4.5, ((x + 0.099) / 1.099) ** (1 / 0.45))


def rec709_oetf(linear: np.ndarray) -> np.ndarray:
    """Linear light -> display-referred Rec.709/BT.709 [0,1]."""
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, None)
    return np.where(x < 0.018, 4.5 * x, 1.099 * np.power(x, 0.45) - 0.099)


def prophoto_eotf(encoded: np.ndarray) -> np.ndarray:
    """Display-referred ProPhoto RGB (ROMM RGB) [0,1] -> linear light.
    Gamma 1.8 with a linear toe below Et = 1/512."""
    x = np.asarray(encoded, dtype=np.float64)
    return np.where(x < 16 * (1 / 512), x / 16.0, x ** 1.8)


def prophoto_oetf(linear: np.ndarray) -> np.ndarray:
    """Linear light -> display-referred ProPhoto RGB [0,1]."""
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, None)
    return np.where(x < (1 / 512), x * 16.0, np.power(x, 1 / 1.8))


def pure_gamma_eotf(encoded: np.ndarray, gamma: float) -> np.ndarray:
    """Display-referred [0,1] -> linear light, simple power law (no toe).
    Used by Adobe RGB (1998), gamma ~= 2.19921875."""
    x = np.clip(np.asarray(encoded, dtype=np.float64), 0.0, None)
    return x ** gamma


def pure_gamma_oetf(linear: np.ndarray, gamma: float) -> np.ndarray:
    """Linear light -> display-referred [0,1], simple power law."""
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, None)
    return x ** (1.0 / gamma)


ADOBE_RGB_GAMMA = 2.19921875


def adobergb_eotf(encoded: np.ndarray) -> np.ndarray:
    return pure_gamma_eotf(encoded, ADOBE_RGB_GAMMA)


def adobergb_oetf(linear: np.ndarray) -> np.ndarray:
    return pure_gamma_oetf(linear, ADOBE_RGB_GAMMA)
