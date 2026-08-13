import cv2
import numpy as np

from core.masking.hue_range import hue_range_mask

# The 8 HSL channels, in wheel order, and the hue (degrees) each is
# centered on - the same non-uniform spacing Lightroom uses (Orange/Purple/
# Magenta aren't evenly spaced from their neighbors because they aren't
# evenly spaced on the actual hue wheel either).
HSL_CHANNELS = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]

CHANNEL_HUE_DEG = {
    "Red": 0.0, "Orange": 30.0, "Yellow": 60.0, "Green": 120.0,
    "Aqua": 180.0, "Blue": 240.0, "Purple": 275.0, "Magenta": 315.0,
}

# Each channel's selection is a smooth hue-distance mask (full strength out
# to CHANNEL_WIDTH_DEG from center, easing to zero over another
# CHANNEL_FEATHER_DEG) rather than a hard-edged hue bin - this is what lets
# adjacent channels (e.g. Orange and Yellow, 30 degrees apart) blend into
# each other instead of showing a seam where one channel's mask ends and
# the next begins.
CHANNEL_WIDTH_DEG = 20.0
CHANNEL_FEATHER_DEG = 20.0

# value=+/-100 on the Hue slider -> +/-60 degrees of hue rotation within
# the channel's mask - enough to visibly shift a channel toward its
# neighbor without being able to rotate it past an adjacent channel.
MAX_HUE_SHIFT_DEG = 60.0

# value=+/-100 on the Luminance slider -> +/-0.5 shift in HSV value (V),
# matching the same 100 -> 1.0-ish scale convention used by the app's other
# +/-100 sliders (Brightness, Highlights, etc).
MAX_LUMINANCE_SHIFT = 0.5


def apply_hsl_grading(image: np.ndarray, hue: dict, saturation: dict, luminance: dict) -> np.ndarray:
    """image: float32 RGB, HxWx3, in [0, 1] (display-referred).
    hue / saturation / luminance: dict of {channel_name: value}, value in
    [-100, 100] (0 or absent = no effect for that channel on that axis).

    Each active channel's Hue/Saturation/Luminance offsets are weighted by
    that channel's own smooth mask and computed from the *original*
    HSV image (not accumulated channel-over-channel), then summed - so the
    result doesn't depend on which order the channels happen to be
    iterated in, and two channels with overlapping masks blend their
    effects additively in the overlap region instead of one clobbering the
    other.
    """
    active_channels = [
        ch for ch in HSL_CHANNELS
        if hue.get(ch) or saturation.get(ch) or luminance.get(ch)
    ]
    if not active_channels:
        return image

    clipped = np.clip(image, 0.0, 1.0).astype(np.float32)
    hsv = cv2.cvtColor(clipped, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    h_delta = np.zeros_like(h)
    s_delta = np.zeros_like(s)
    v_delta = np.zeros_like(v)

    for ch in active_channels:
        mask = hue_range_mask(h, CHANNEL_HUE_DEG[ch], CHANNEL_WIDTH_DEG, CHANNEL_FEATHER_DEG)

        hue_value = hue.get(ch, 0)
        if hue_value:
            h_delta += mask * (hue_value / 100.0) * MAX_HUE_SHIFT_DEG

        sat_value = saturation.get(ch, 0)
        if sat_value:
            # Same convention as the global Saturation tool
            # (factor = 1 + value/100), restricted to this channel's mask.
            s_delta += mask * s * (sat_value / 100.0)

        luma_value = luminance.get(ch, 0)
        if luma_value:
            v_delta += mask * (luma_value / 100.0) * MAX_LUMINANCE_SHIFT

    hsv_out = np.stack([
        (h + h_delta) % 360.0,
        np.clip(s + s_delta, 0.0, 1.0),
        np.clip(v + v_delta, 0.0, 1.0),
    ], axis=-1).astype(np.float32)

    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)
