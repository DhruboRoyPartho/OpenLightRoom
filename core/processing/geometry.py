import math
import numpy as np
import cv2


def rotate90(image: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Rotate by a multiple of 90 degrees clockwise. Lossless - no
    interpolation or cropping involved."""
    q = quarter_turns % 4
    if q == 0:
        return image
    if q == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if q == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def flip(image: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    if not horizontal and not vertical:
        return image
    if horizontal and vertical:
        return cv2.flip(image, -1)
    if horizontal:
        return cv2.flip(image, 1)
    return cv2.flip(image, 0)


def rotate_rect_90(rect, clockwise: bool):
    """Transform a normalized (x0, y0, x1, y1) rect the same way rotate90()
    would transform the pixels it bounds, so a crop rectangle keeps
    selecting the same visual content after the frame is rotated."""
    x0, y0, x1, y1 = rect
    if clockwise:
        return (1 - y1, x0, 1 - y0, x1)
    return (y0, 1 - x1, y1, 1 - x0)


def flip_rect(rect, horizontal: bool, vertical: bool):
    """Mirror of rotate_rect_90 for flip(), keeping a crop rectangle
    pinned to the same visual content after a flip."""
    x0, y0, x1, y1 = rect
    if horizontal:
        x0, x1 = 1 - x1, 1 - x0
    if vertical:
        y0, y1 = 1 - y1, 1 - y0
    return (x0, y0, x1, y1)


def _largest_inscribed_rect(w: float, h: float, angle_rad: float):
    """Width/height of the largest axis-aligned rectangle that fits inside a
    w x h rectangle after it's been rotated by angle_rad, without touching
    the empty corners the rotation opens up. Standard closed-form solution
    for the "rotate an image and crop out the black borders" problem."""
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    sin_a, cos_a = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # Half-constrained: two crop corners touch the long side, the other
        # two sit on the midline.
        x = 0.5 * side_short
        if width_is_longer:
            wr, hr = x / sin_a, x / cos_a
        else:
            wr, hr = x / cos_a, x / sin_a
    else:
        # Fully constrained: the crop touches all four sides.
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def _inscribed_crop_size(w: float, h: float, angle_rad: float, max_w: int, max_h: int):
    """_largest_inscribed_rect(), clamped to the available canvas and inset
    by a couple of pixels: the rotated content's true edge is anti-aliased
    by INTER_LINEAR, so a crop touching the exact theoretical boundary can
    pick up a sliver of that blended edge."""
    crop_w, crop_h = _largest_inscribed_rect(w, h, angle_rad)
    inset = 2
    crop_w = max(1, min(int(crop_w) - inset, max_w))
    crop_h = max(1, min(int(crop_h) - inset, max_h))
    return crop_w, crop_h


def rotate_arbitrary(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Straighten: rotate by an arbitrary angle around the center and crop to
    the largest rectangle that avoids the empty corners the rotation opens
    up, so the result is a clean full rectangle with no black triangles."""
    if not angle_degrees:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    angle_rad = math.radians(angle_degrees)

    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(round(h * sin_a + w * cos_a))
    new_h = int(round(h * cos_a + w * sin_a))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    crop_w, crop_h = _inscribed_crop_size(w, h, angle_rad, new_w, new_h)
    x0 = (new_w - crop_w) // 2
    y0 = (new_h - crop_h) // 2

    return rotated[y0:y0 + crop_h, x0:x0 + crop_w]


def straighten_angle_from_line(dx: float, dy: float) -> float:
    """Given a line the user dragged across the image (dx, dy in pixels,
    y increasing downward), return the straighten angle (degrees, matching
    rotate_arbitrary's sign convention) that makes that line horizontal -
    i.e. what dragging along a tilted horizon should produce."""
    if dx == 0 and dy == 0:
        return 0.0
    screen_angle = math.degrees(math.atan2(dy, dx))
    # Fold to (-90, 90] so a near-vertical drag is read as "the short way"
    # to horizontal.
    return ((screen_angle + 90) % 180) - 90


def downscale_to_max_dimension(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """Scales image down (never up) so its longer side is at most
    max_dimension, preserving aspect ratio, via INTER_AREA - the correct
    OpenCV filter for shrinking, since it averages every source pixel that
    lands in each output pixel rather than aliasing fine detail away the
    way INTER_LINEAR/NEAREST would at a large size reduction.

    Used only for the interactive preview render (see ImageDocument.render
    / RenderQueue's preview_max_dimension): a smaller working image makes
    every per-pixel adjustment in the color pipeline proportionally
    faster. Never used for the export/final-output path, which always
    renders the untouched base_image at full resolution.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if max_dimension is None or longest <= max_dimension:
        return image
    scale = max_dimension / float(longest)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_normalized(image: np.ndarray, rect) -> np.ndarray:
    """rect: (x0, y0, x1, y1) as fractions of width/height, each in [0, 1].
    A no-op is returned unchanged (no copy) if rect covers the full frame."""
    x0, y0, x1, y1 = rect
    h, w = image.shape[:2]

    left = int(round(x0 * w))
    top = int(round(y0 * h))
    right = int(round(x1 * w))
    bottom = int(round(y1 * h))

    left = max(0, min(left, w - 1))
    top = max(0, min(top, h - 1))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))

    if left == 0 and top == 0 and right == w and bottom == h:
        return image

    return image[top:bottom, left:right].copy()
