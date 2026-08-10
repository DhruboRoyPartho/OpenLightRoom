import numpy as np

IDENTITY_POINTS = [(0, 0), (255, 255)]


def _sample_spline(points, sample_x: np.ndarray) -> np.ndarray:
    """Evaluate a Catmull-Rom spline through points (sorted by x) at the
    given sample_x array, returning y values clipped to [0, 255]. Shared by
    build_lut() (256-entry table for the on-screen curve preview) and
    apply_curve() (a finer-resolution table used for actual pixel
    processing, so curve edits don't introduce 8-bit banding into an
    otherwise float32 pipeline)."""
    pts = sorted(points, key=lambda p: p[0])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)

    if len(pts) < 2:
        return np.clip(sample_x.astype(np.float64), 0, 255)

    if len(pts) == 2:
        return np.clip(np.interp(sample_x, xs, ys), 0, 255)

    # Catmull-Rom needs one "phantom" point before the first and after the
    # last control point so the end segments have a defined tangent.
    ext_xs = np.concatenate(([xs[0] - (xs[1] - xs[0])], xs, [xs[-1] + (xs[-1] - xs[-2])]))
    ext_ys = np.concatenate(([ys[0] - (ys[1] - ys[0])], ys, [ys[-1] + (ys[-1] - ys[-2])]))

    lut = sample_x.astype(np.float64).copy()

    for i in range(1, len(ext_xs) - 2):
        x0, x1, x2, x3 = ext_xs[i - 1:i + 3]
        y0, y1, y2, y3 = ext_ys[i - 1:i + 3]
        if x2 <= x1:
            continue

        mask = (sample_x >= x1) & (sample_x <= x2)
        if not np.any(mask):
            continue

        t = (sample_x[mask] - x1) / (x2 - x1)
        t2 = t * t
        t3 = t2 * t

        a0 = -0.5 * t3 + t2 - 0.5 * t
        a1 = 1.5 * t3 - 2.5 * t2 + 1.0
        a2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
        a3 = 0.5 * t3 - 0.5 * t2

        lut[mask] = a0 * y0 + a1 * y1 + a2 * y2 + a3 * y3

    return np.clip(lut, 0, 255)


def build_lut(points) -> np.ndarray:
    """256-entry curve sampled at integer x=0..255, used by the curve
    widget's on-screen preview."""
    return _sample_spline(points, np.arange(256, dtype=np.float64))


def apply_curve(image: np.ndarray, points_by_channel: dict) -> np.ndarray:
    """Apply an RGB master curve followed by independent per-channel
    curves, matching Lightroom's Point Curve channel model. image: float32
    display-referred data (not assumed pre-clipped to [0,1], since earlier
    display-stage tools in the chain may have pushed values outside that
    range before the final clip). Uses np.interp against a fine-resolution
    spline sampling rather than cv2.LUT (which requires uint8), so curve
    edits don't quantize the float pipeline down to 8-bit steps."""
    sample_x = np.linspace(0.0, 255.0, 512)
    result = image * np.float32(255.0)

    master_pts = points_by_channel.get("RGB", IDENTITY_POINTS)
    if list(master_pts) != list(IDENTITY_POINTS):
        master_y = _sample_spline(master_pts, sample_x)
        result = np.interp(result, sample_x, master_y)

    channel_index = {"Red": 0, "Green": 1, "Blue": 2}
    for name, idx in channel_index.items():
        pts = points_by_channel.get(name, IDENTITY_POINTS)
        if list(pts) == list(IDENTITY_POINTS):
            continue
        y = _sample_spline(pts, sample_x)
        result[:, :, idx] = np.interp(result[:, :, idx], sample_x, y)

    return (result / 255.0).astype(np.float32)
