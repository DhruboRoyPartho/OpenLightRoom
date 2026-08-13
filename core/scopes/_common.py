import numpy as np


def clean_image(image: np.ndarray) -> np.ndarray:
    """NaN/Inf-safe, [0, 1]-clamped copy of a rendered image - scopes must
    never raise or produce garbage bins just because a pixel is transiently
    out-of-range mid-edit; they fail safe by treating NaN as black and
    Inf/-Inf as white/black rather than propagating them into a histogram
    bin index."""
    return np.clip(np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def column_bin_indices(width: int, out_width: int, height: int) -> np.ndarray:
    """An (height, width) int array mapping each pixel's column to one of
    out_width output columns, for the waveform/parade's per-column
    histogram - built once per width, broadcast down every row."""
    xs = np.arange(width, dtype=np.float64)
    col_idx = np.floor(xs / max(width, 1) * out_width).astype(np.int64)
    col_idx = np.clip(col_idx, 0, out_width - 1)
    return np.broadcast_to(col_idx, (height, width))
