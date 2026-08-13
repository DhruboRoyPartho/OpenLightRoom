"""Reusable, UI-independent scope generation: pure numpy functions that
turn a rendered (display-referred) image into the data a Histogram,
Waveform, RGB Parade, or Vectorscope draws. Every function here is fully
vectorized (np.histogram / np.histogram2d) - no per-pixel Python loops -
so these stay fast at full working resolution (24-60MP+) and are
independently unit-testable without any Qt/UI dependency.
"""

from core.scopes.histogram import compute_rgb_histogram, compute_luminance_histogram
from core.scopes.waveform import compute_waveform
from core.scopes.rgb_parade import compute_rgb_parade
from core.scopes.vectorscope import compute_vectorscope, SKIN_TONE_ANGLE_DEG

__all__ = [
    "compute_rgb_histogram", "compute_luminance_histogram",
    "compute_waveform",
    "compute_rgb_parade",
    "compute_vectorscope", "SKIN_TONE_ANGLE_DEG",
]
