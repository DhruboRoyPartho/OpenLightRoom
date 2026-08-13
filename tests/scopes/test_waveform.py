"""Tests for core/scopes/waveform.py."""

import numpy as np

from core.scopes.waveform import compute_waveform


def test_solid_gray_image_produces_a_single_bright_row():
    image = np.full((20, 40, 3), 0.5, dtype=np.float32)
    wf = compute_waveform(image, out_width=40, out_height=100)
    assert wf.shape == (100, 40)
    # luma=0.5 -> row index (100-1) - int(0.5*100) roughly the middle row;
    # every count should land in exactly one row per column.
    nonzero_rows = np.nonzero(wf.sum(axis=1))[0]
    assert len(nonzero_rows) == 1
    assert wf.sum() == 20 * 40


def test_black_left_half_white_right_half_separates_by_column():
    image = np.zeros((10, 20, 3), dtype=np.float32)
    image[:, 10:, :] = 1.0
    wf = compute_waveform(image, out_width=20, out_height=50)

    left_columns = wf[:, :10]
    right_columns = wf[:, 10:]
    # Black -> luma 0 -> bottom row (index out_height-1); white -> top row (index 0).
    assert left_columns[-1, :].sum() == 10 * 10   # 10 rows x 10 black columns
    assert right_columns[0, :].sum() == 10 * 10


def test_total_density_equals_pixel_count():
    rng = np.random.default_rng(0)
    image = rng.random((15, 30, 3)).astype(np.float32)
    wf = compute_waveform(image, out_width=64, out_height=64)
    assert wf.sum() == 15 * 30


def test_output_shape_matches_requested_dimensions():
    image = np.zeros((5, 5, 3), dtype=np.float32)
    wf = compute_waveform(image, out_width=128, out_height=64)
    assert wf.shape == (64, 128)


def test_nan_and_out_of_gamut_input_do_not_raise():
    weird = np.array([[[np.nan, 0.5, 0.5], [np.inf, -np.inf, 2.0]]], dtype=np.float32)
    wf = compute_waveform(weird, out_width=8, out_height=8)
    assert wf.shape == (8, 8)
    assert np.isfinite(wf).all()
