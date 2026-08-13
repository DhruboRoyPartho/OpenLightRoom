import numpy as np

from core.color_science.oklab import linear_srgb_to_oklab, oklab_to_linear_srgb
from core.processing.color_space import srgb_to_linear, linear_to_srgb
from core.masking.luminance_range import three_zone_luminance_masks

# chroma=100 (the wheel's outer edge) -> this much OKLab chroma pushed into
# the selected tonal range. OKLab chroma for in-gamut sRGB colors tops out
# around ~0.3-0.4 at moderate lightness, so 0.28 is a strong-but-not-gamut-
# absurd push at full radius.
MAX_CHROMA = 0.28

# luminance=100 -> this much added to OKLab L (which itself runs roughly
# 0..1 for sRGB black..white) - a moderate brightness push per wheel,
# smaller than the chroma push since OKLab L is far more visually
# sensitive per-unit than a/b.
MAX_LUMINANCE_OFFSET = 0.15

DEFAULT_WHEEL = {"hue_deg": 0.0, "chroma": 0.0, "luminance": 0.0}
ZONES = ("shadows", "midtones", "highlights", "global")


def _is_active(wheel: dict) -> bool:
    return bool(wheel) and (wheel.get("chroma", 0.0) or wheel.get("luminance", 0.0))


def _hue_chroma_to_ab(hue_deg: float, chroma_pct: float):
    theta = np.deg2rad(hue_deg)
    chroma = (chroma_pct / 100.0) * MAX_CHROMA
    return chroma * np.cos(theta), chroma * np.sin(theta)


def apply_color_wheels(image: np.ndarray, shadows: dict, midtones: dict,
                        highlights: dict, global_: dict) -> np.ndarray:
    """image: float32 RGB, HxWx3, in [0, 1] (display-referred / gamma-
    encoded sRGB). Each of shadows/midtones/highlights/global_ is a dict
    with 'hue_deg' (0..360), 'chroma' (0..100, wheel-radius percentage) and
    'luminance' (-100..100); an empty/all-zero dict skips that wheel.

    The push is applied in OKLab - a perceptually uniform space - by
    offsetting a/b toward the wheel's hue at a magnitude set by its
    chroma, and offsetting L by its luminance. Shadows/Midtones/Highlights
    are weighted by the same three-zone, partition-of-unity luminance mask
    used by tonal separation elsewhere in the app (see
    core/masking/luminance_range.py), so the three zones blend smoothly
    into each other instead of showing a hard seam; Global applies at full
    strength everywhere and stacks on top of whichever zone(s) a pixel
    falls into.
    """
    wheels = {"shadows": shadows, "midtones": midtones, "highlights": highlights, "global": global_}
    if not any(_is_active(w) for w in wheels.values()):
        return image

    clipped = np.clip(image, 0.0, 1.0).astype(np.float32)
    linear = srgb_to_linear(clipped)
    lab = linear_srgb_to_oklab(linear)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    L_delta = np.zeros_like(L)
    a_delta = np.zeros_like(a)
    b_delta = np.zeros_like(b)

    if _is_active(wheels["global"]):
        w = wheels["global"]
        da, db = _hue_chroma_to_ab(w.get("hue_deg", 0.0), w.get("chroma", 0.0))
        a_delta += da
        b_delta += db
        L_delta += (w.get("luminance", 0.0) / 100.0) * MAX_LUMINANCE_OFFSET

    if _is_active(wheels["shadows"]) or _is_active(wheels["midtones"]) or _is_active(wheels["highlights"]):
        shadows_mask, midtones_mask, highlights_mask = three_zone_luminance_masks(L)
        for mask, key in ((shadows_mask, "shadows"), (midtones_mask, "midtones"), (highlights_mask, "highlights")):
            w = wheels[key]
            if not _is_active(w):
                continue
            da, db = _hue_chroma_to_ab(w.get("hue_deg", 0.0), w.get("chroma", 0.0))
            a_delta += mask * da
            b_delta += mask * db
            L_delta += mask * (w.get("luminance", 0.0) / 100.0) * MAX_LUMINANCE_OFFSET

    lab_out = np.stack([L + L_delta, a + a_delta, b + b_delta], axis=-1).astype(np.float32)
    linear_out = oklab_to_linear_srgb(lab_out)
    return linear_to_srgb(np.clip(linear_out, 0.0, None))
