import numpy as np

from core.scopes._common import clean_image

# Rec. 601 RGB -> Cb/Cr (chroma-only, Y dropped) - the classic broadcast
# vectorscope's color-difference plane. Cb/Cr each range [-0.5, 0.5]
# for in-gamut RGB; a neutral gray always lands at the origin regardless
# of its brightness, which is what makes a vectorscope a pure "how far off
# neutral, and toward which hue" instrument.
_CB_COEFFS = np.array([-0.168736, -0.331264, 0.5])
_CR_COEFFS = np.array([0.5, -0.418688, -0.081312])

# The traditional NTSC vectorscope "skin tone line" / I-line: healthy skin
# tones of any ethnicity cluster tightly along this angle (measured
# counter-clockwise from the +Cb axis in the Cb/Cr plane) regardless of how
# saturated or desaturated they are - graders use it as a reference for
# whether skin tones have picked up an unwanted color cast.
SKIN_TONE_ANGLE_DEG = 123.0


def rgb_to_cb_cr(image: np.ndarray):
    """image: float32 RGB, ...x3, in [0, 1]. Returns (cb, cr), each the
    same leading shape as image (channel axis dropped), in [-0.5, 0.5]."""
    clipped = clean_image(image)
    cb = clipped @ _CB_COEFFS
    cr = clipped @ _CR_COEFFS
    return cb, cr


def cb_cr_to_hue_angle_deg(cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """The angle (degrees, [0, 360)) of a Cb/Cr point around the origin -
    what SKIN_TONE_ANGLE_DEG is measured in, and what a UI draws hue
    targets (red/yellow/green/cyan/blue/magenta) at around the scope."""
    return np.degrees(np.arctan2(cr, cb)) % 360.0


def compute_vectorscope(image: np.ndarray, size: int = 256) -> np.ndarray:
    """image: float32 RGB, HxWx3, in [0, 1]. Returns a (size, size)
    float64 density map of every pixel's Cb/Cr position - a neutral image
    is a single bright spot at the center; a strong color cast or heavily
    saturated content pushes the density outward and around, toward that
    color's hue angle. Built with one np.histogram2d call over the whole
    image, not a per-pixel Python loop.
    """
    cb, cr = rgb_to_cb_cr(image)
    hist2d, _, _ = np.histogram2d(
        cb.ravel(), cr.ravel(),
        bins=[size, size], range=[[-0.5, 0.5], [-0.5, 0.5]],
    )
    # Cr (red-difference) increasing should read upward, matching every
    # broadcast vectorscope's convention, so flip vertically.
    return hist2d.T[::-1, :]
