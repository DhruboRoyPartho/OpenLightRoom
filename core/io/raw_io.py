import os
import numpy as np
import rawpy

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}


def is_raw_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS


def load_raw(path: str) -> np.ndarray:
    """Decode a camera RAW file into a scene-linear float32 RGB array via
    LibRaw: real demosaicing, the camera's as-shot white balance, and
    LibRaw's built-in color matrix for that camera model - the standard
    "camera profile" baseline every non-Adobe raw workflow uses (a custom,
    per-camera calibrated DCP-style profile is a further step this doesn't
    attempt). gamma=(1, 1) means no tone curve is applied by LibRaw, so the
    result is linear light, ready to feed straight into this app's linear
    working stage - the same role a decoded-and-linearized JPEG/PNG plays.
    """
    with rawpy.imread(path) as raw:
        rgb16 = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
        )
    return rgb16.astype(np.float32) / 65535.0
