"""Tests for the pure numpy->QImage helpers in interface/gui/scopes_panel.py
(the "basic display" conversion for Waveform/RGB Parade/Vectorscope)."""

import numpy as np
import pytest
from PySide6.QtGui import QColor

from interface.gui.scopes_panel import _normalize_to_u8, _density_to_qimage, _parade_to_qimage


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_normalize_maps_all_zero_data_to_all_zero_output():
    data = np.zeros((10, 10))
    out = _normalize_to_u8(data)
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_normalize_peak_bin_maps_to_max_brightness():
    data = np.zeros((5, 5))
    data[2, 2] = 1000.0
    out = _normalize_to_u8(data)
    assert out[2, 2] == 255
    assert out.max() == 255


def test_normalize_output_always_in_uint8_range():
    rng = np.random.default_rng(0)
    data = rng.random((8, 8)) * 1e6
    out = _normalize_to_u8(data)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_density_to_qimage_grayscale_has_correct_size(app):
    data = np.random.default_rng(0).random((20, 40)) * 100
    qimg = _density_to_qimage(data)
    assert qimg.width() == 40
    assert qimg.height() == 20


def test_density_to_qimage_tint_colors_the_output(app):
    data = np.zeros((4, 4))
    data[1, 1] = 100.0
    qimg = _density_to_qimage(data, tint=QColor(255, 0, 0))
    color = qimg.pixelColor(1, 1)
    assert color.red() > 0
    assert color.green() == 0
    assert color.blue() == 0


def test_parade_to_qimage_channels_map_to_correct_rgb_planes(app):
    h, w = 4, 4
    parade = {
        "R": np.full((h, w), 1000.0),
        "G": np.zeros((h, w)),
        "B": np.zeros((h, w)),
    }
    qimg = _parade_to_qimage(parade)
    color = qimg.pixelColor(0, 0)
    assert color.red() == 255
    assert color.green() == 0
    assert color.blue() == 0


def test_parade_to_qimage_size_matches_input(app):
    parade = {ch: np.zeros((12, 30)) for ch in ("R", "G", "B")}
    qimg = _parade_to_qimage(parade)
    assert qimg.width() == 30
    assert qimg.height() == 12
