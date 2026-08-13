"""Tests for core/io/preset_io.py. Each test points PRESETS_DIR at a fresh
pytest tmp_path so these never touch the app's real presets/ folder."""

import os
import pytest

from core.io import preset_io
from core.adjustment_layers.exposure_layer import ExposureLayer
from core.adjustment_layers.saturation_layer import SaturationLayer
from core.adjustment_layers.geometry_layer import GeometryLayer
from core.adjustment_layers.hsl_layer import HSLLayer


@pytest.fixture(autouse=True)
def isolated_presets_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(preset_io, "PRESETS_DIR", str(tmp_path / "presets"))
    yield tmp_path


def _some_layers():
    return [ExposureLayer(20.0), SaturationLayer(-15.0)]


def test_list_presets_empty_when_no_presets_saved():
    assert preset_io.list_presets() == []


def test_save_and_load_round_trips_layer_values():
    preset_io.save_preset("Moody", _some_layers())
    assert preset_io.preset_exists("Moody")
    assert "Moody" in preset_io.list_presets()

    loaded = preset_io.load_preset("Moody")
    by_name = {str(l): l for l in loaded}
    assert set(by_name) == {"Exposure", "Saturation"}
    assert by_name["Exposure"].exposure_factor == 20.0
    assert by_name["Saturation"].saturation_value == -15.0


def test_save_without_overwrite_raises_if_already_exists():
    preset_io.save_preset("Moody", _some_layers())
    with pytest.raises(FileExistsError):
        preset_io.save_preset("Moody", _some_layers())


def test_save_with_overwrite_replaces_existing():
    preset_io.save_preset("Moody", [ExposureLayer(10.0)])
    preset_io.save_preset("Moody", [ExposureLayer(50.0)], overwrite=True)
    loaded = preset_io.load_preset("Moody")
    assert loaded[0].exposure_factor == 50.0


def test_crop_layer_is_excluded_from_saved_presets():
    layers = [ExposureLayer(10.0), GeometryLayer(crop_rect=(0.1, 0.1, 0.9, 0.9))]
    preset_io.save_preset("NoGeometry", layers)
    loaded = preset_io.load_preset("NoGeometry")
    names = {str(l) for l in loaded}
    assert names == {"Exposure"}


def test_multi_field_layer_round_trips_through_a_preset():
    layers = [HSLLayer(hue={"Red": 15}, saturation={"Blue": -20})]
    preset_io.save_preset("HSL Look", layers)
    loaded = preset_io.load_preset("HSL Look")
    assert len(loaded) == 1
    assert loaded[0].hue == {"Red": 15}
    assert loaded[0].saturation == {"Blue": -20}


def test_load_missing_preset_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        preset_io.load_preset("DoesNotExist")


def test_delete_preset_removes_it():
    preset_io.save_preset("Temp", _some_layers())
    preset_io.delete_preset("Temp")
    assert not preset_io.preset_exists("Temp")
    assert "Temp" not in preset_io.list_presets()


def test_delete_missing_preset_is_a_silent_no_op():
    preset_io.delete_preset("NeverExisted")  # should not raise


def test_duplicate_preset_creates_an_independent_copy():
    preset_io.save_preset("Original", _some_layers())
    preset_io.duplicate_preset("Original", "Copy")

    assert set(preset_io.list_presets()) == {"Original", "Copy"}
    original = preset_io.load_preset("Original")
    copy = preset_io.load_preset("Copy")
    assert {str(l) for l in original} == {str(l) for l in copy}

    # Independent: overwriting one doesn't touch the other.
    preset_io.save_preset("Copy", [ExposureLayer(99.0)], overwrite=True)
    assert preset_io.load_preset("Original")[0].exposure_factor == 20.0


def test_duplicate_missing_source_raises():
    with pytest.raises(FileNotFoundError):
        preset_io.duplicate_preset("Nope", "AlsoNope")


def test_export_and_import_round_trip(tmp_path):
    preset_io.save_preset("Exportable", _some_layers())
    export_path = str(tmp_path / "shared_preset.json")
    preset_io.export_preset("Exportable", export_path)
    assert os.path.isfile(export_path)

    preset_io.delete_preset("Exportable")
    assert "Exportable" not in preset_io.list_presets()

    imported_name = preset_io.import_preset(export_path)
    assert imported_name == "Exportable"
    assert "Exportable" in preset_io.list_presets()
    loaded = preset_io.load_preset("Exportable")
    assert {str(l) for l in loaded} == {"Exposure", "Saturation"}


def test_import_with_explicit_name_override(tmp_path):
    preset_io.save_preset("Original Name", _some_layers())
    export_path = str(tmp_path / "shared.json")
    preset_io.export_preset("Original Name", export_path)

    imported_name = preset_io.import_preset(export_path, name="Renamed On Import")
    assert imported_name == "Renamed On Import"
    assert "Renamed On Import" in preset_io.list_presets()


def test_preset_name_with_path_unsafe_characters_is_sanitized():
    preset_io.save_preset("A/B:C?D", _some_layers())
    assert any(preset_io.list_presets())  # didn't raise, produced a file somewhere sane
    # The sanitized name should be loadable back via the same original name.
    loaded = preset_io.load_preset("A/B:C?D")
    assert {str(l) for l in loaded} == {"Exposure", "Saturation"}


def test_empty_preset_name_raises():
    with pytest.raises(ValueError):
        preset_io.save_preset("   ", _some_layers())
