import numpy as np

_EPS = 1e-6


def _solve_temp_tint(avg_r: float, avg_g: float, avg_b: float):
    """Given a scene-linear RGB triple that should end up neutral gray
    after correction, solve for the Temperature/Tint slider values (each
    in the app's -100..100 convention, see core/processing/temperature.py
    and tint.py) that make it so.

    The app's Temperature/Tint layers apply:
        R' = R * 2**(temp/100)
        G' = G * 2**(-tint/100)
        B' = B / 2**(temp/100)
    Setting R' == G' == B' and solving for temp/tint gives a closed form -
    no iterative search needed, unlike most naive gray-world
    implementations.
    """
    r = max(avg_r, _EPS)
    g = max(avg_g, _EPS)
    b = max(avg_b, _EPS)

    temp_factor = float(np.sqrt(b / r))
    target = r * temp_factor
    tint_factor = target / g

    temp = 100.0 * np.log2(temp_factor)
    tint = -100.0 * np.log2(tint_factor)

    return float(np.clip(temp, -100.0, 100.0)), float(np.clip(tint, -100.0, 100.0))


def estimate_gray_world_white_balance(image: np.ndarray):
    """image: float32 RGB, HxWx3, scene-linear (the working image before
    Temperature/Tint are applied - i.e. what document.base_image, after
    Crop, looks like). Gray-world assumption: a natural scene's average
    color, across all its content, should be neutral gray - the standard
    basis for "Auto White Balance".

    Returns (temperature_value, tint_value) in the app's -100..100 slider
    convention: the pair that, applied as Temperature/Tint, makes the
    image's average color neutral.
    """
    clipped = np.clip(image, 0.0, None)
    avg_r = float(np.mean(clipped[..., 0]))
    avg_g = float(np.mean(clipped[..., 1]))
    avg_b = float(np.mean(clipped[..., 2]))
    return _solve_temp_tint(avg_r, avg_g, avg_b)


def estimate_white_balance_from_sample(r: float, g: float, b: float):
    """r/g/b: a single scene-linear RGB sample (e.g. an eyedropper click)
    that the user has identified as "should be neutral gray" - a gray
    card, a white wall, a neutral sweater. Returns (temperature_value,
    tint_value) in the same convention as
    estimate_gray_world_white_balance."""
    return _solve_temp_tint(max(r, 0.0), max(g, 0.0), max(b, 0.0))


def sample_scene_linear_pixel(document, px: int, py: int):
    """document: anything with .base_image (scene-linear HxWx3) and
    .layers (a list whose items stringify to a layer name, one of which
    may be "Crop"). px, py: pixel coordinates in the document's rendered
    (post-crop) frame - the same coordinate space ImageDocument.render()
    produces, which is what the on-screen canvas displays.

    Returns the scene-linear (r, g, b) at that pixel with only geometry
    (crop/rotate/flip/straighten) applied - i.e. before Temperature/Tint/
    Exposure/etc - so an eyedropper reading is independent of whatever
    White Balance is currently dialed in (clicking the same physical spot
    always measures the same underlying color, regardless of prior edits).
    """
    crop_layer = next((l for l in document.layers if str(l) == "Crop"), None)
    image = document.base_image
    if crop_layer is not None:
        image = crop_layer.apply(image)

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return 0.0, 0.0, 0.0
    px = max(0, min(w - 1, int(px)))
    py = max(0, min(h - 1, int(py)))
    r, g, b = image[py, px]
    return float(r), float(g), float(b)
