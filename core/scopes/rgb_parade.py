import numpy as np

from core.scopes._common import clean_image, column_bin_indices

CHANNELS = ("R", "G", "B")


def compute_rgb_parade(image: np.ndarray, out_width: int = 256, out_height: int = 256) -> dict:
    """Like compute_waveform, but per channel: {"R": density, "G": ...,
    "B": ...}, each an (out_height, out_width) float64 map showing that one
    channel's per-column distribution - the standard "RGB Parade" scope,
    which is a waveform run three times (once per channel) so channel
    imbalances (a color cast, a clipped channel) are visible independently
    instead of averaged away into luma.
    """
    clipped = clean_image(image)
    h, w = clipped.shape[:2]
    col_idx = column_bin_indices(w, out_width, h)
    col_flat = col_idx.ravel()

    result = {}
    for i, ch in enumerate(CHANNELS):
        channel = clipped[..., i]
        hist2d, _, _ = np.histogram2d(
            col_flat, channel.ravel(),
            bins=[out_width, out_height], range=[[0, out_width], [0.0, 1.0]],
        )
        result[ch] = hist2d.T[::-1, :]
    return result
