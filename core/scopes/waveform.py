import numpy as np

from core.scopes._common import clean_image, column_bin_indices
from core.processing.color_space import luminance


def compute_waveform(image: np.ndarray, out_width: int = 256, out_height: int = 256) -> np.ndarray:
    """image: float32 RGB, HxWx3, in [0, 1]. Returns an (out_height,
    out_width) float64 density map: column x is a per-column histogram of
    that column's luma values, so a waveform monitor's classic "how bright
    is each part of the frame, left to right" reading falls out directly.
    Row 0 is luma=1.0 (brightest, drawn at the top); row out_height-1 is
    luma=0.0. Built with a single np.histogram2d call over every pixel at
    once - not a per-column or per-pixel Python loop - so this stays fast
    at full working resolution.
    """
    clipped = clean_image(image)
    h, w = clipped.shape[:2]
    luma = luminance(clipped)

    col_idx = column_bin_indices(w, out_width, h)
    hist2d, _, _ = np.histogram2d(
        col_idx.ravel(), luma.ravel(),
        bins=[out_width, out_height], range=[[0, out_width], [0.0, 1.0]],
    )
    # hist2d is (out_width, out_height) with luma increasing along axis 1;
    # transpose to (height, width) image orientation and flip vertically
    # so brighter luma renders at the top, matching every waveform monitor.
    return hist2d.T[::-1, :]
