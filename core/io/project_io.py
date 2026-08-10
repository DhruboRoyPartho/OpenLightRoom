import json

from core.image_model.image_document import ImageDocument
from core.io.image_io import load_and_linearize
from core.io.raw_io import is_raw_file, load_raw
from core.io.exif_io import read_exif
from core.adjustment_layers.brightness_layer import BrightnessLayer
from core.adjustment_layers.contrast_layer import ContrastLayer
from core.adjustment_layers.temperature_layer import TemperatureLayer
from core.adjustment_layers.tint_layer import TintLayer
from core.adjustment_layers.exposure_layer import ExposureLayer
from core.adjustment_layers.highlights_layer import HighlightsLayer
from core.adjustment_layers.shadows_layer import ShadowsLayer
from core.adjustment_layers.whites_layer import WhitesLayer
from core.adjustment_layers.blacks_layer import BlacksLayer
from core.adjustment_layers.curve_layer import CurveLayer
from core.adjustment_layers.geometry_layer import GeometryLayer
from core.adjustment_layers.vibrance_layer import VibranceLayer
from core.adjustment_layers.saturation_layer import SaturationLayer
from core.adjustment_layers.hue_layer import HueLayer
from core.adjustment_layers.parametric_curve_layer import ParametricCurveLayer

# A project file is a small JSON "sidecar": it records which source image it
# edits and the current value of each adjustment layer, not pixel data. This
# mirrors the app's own non-destructive model - the source image is re-read
# from disk and the layers are re-applied on top of it when reopened.
PROJECT_VERSION = 1

# Every one of these layers has a single scalar constructor argument, so
# they can be saved/loaded generically as {"type": name, "value": x}. Curve
# is handled separately below since it carries per-channel point lists.
LAYER_TYPES = {
    "Brightness": BrightnessLayer,
    "Contrast": ContrastLayer,
    "Temperature": TemperatureLayer,
    "Tint": TintLayer,
    "Exposure": ExposureLayer,
    "Highlights": HighlightsLayer,
    "Shadows": ShadowsLayer,
    "Whites": WhitesLayer,
    "Blacks": BlacksLayer,
    "Vibrance": VibranceLayer,
    "Saturation": SaturationLayer,
    "Hue": HueLayer,
}


def _serialize_layer(layer):
    name = str(layer)
    if name == "Curve":
        return {"type": name, "points_by_channel": layer.points_by_channel}
    if name == "Parametric Curve":
        return {
            "type": name,
            "highlights": layer.highlights,
            "lights": layer.lights,
            "darks": layer.darks,
            "shadows": layer.shadows,
        }
    if name == "Crop":
        return {
            "type": name,
            "rotation90": layer.rotation90,
            "flip_h": layer.flip_h,
            "flip_v": layer.flip_v,
            "crop_rect": list(layer.crop_rect),
            "angle": layer.angle,
        }
    return {"type": name, "value": next(iter(vars(layer).values()))}


def save_project(path: str, image_path: str, document: ImageDocument):
    data = {
        "version": PROJECT_VERSION,
        "image_path": image_path,
        "layers": [_serialize_layer(layer) for layer in document.layers],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_project(path: str):
    """Returns (image_path, ImageDocument). Raises ValueError/FileNotFoundError
    if the project file or its referenced source image can't be read."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = data.get("image_path")
    if not image_path:
        raise ValueError("Project file is missing its source image path.")

    try:
        image = load_raw(image_path) if is_raw_file(image_path) else load_and_linearize(image_path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise FileNotFoundError(f"Could not read the source image:\n{image_path}\n({e})")

    document = ImageDocument(image)
    document.exif_data = read_exif(image_path)
    for entry in data.get("layers", []):
        layer_type = entry.get("type")
        if layer_type == "Curve":
            document.layers.append(CurveLayer(entry.get("points_by_channel")))
            continue
        if layer_type == "Parametric Curve":
            document.layers.append(ParametricCurveLayer(
                highlights=entry.get("highlights", 0),
                lights=entry.get("lights", 0),
                darks=entry.get("darks", 0),
                shadows=entry.get("shadows", 0),
            ))
            continue
        if layer_type == "Crop":
            document.layers.append(GeometryLayer(
                rotation90=entry.get("rotation90", 0),
                flip_h=entry.get("flip_h", False),
                flip_v=entry.get("flip_v", False),
                crop_rect=tuple(entry.get("crop_rect")) if entry.get("crop_rect") else None,
                angle=entry.get("angle", 0.0),
            ))
            continue
        layer_cls = LAYER_TYPES.get(layer_type)
        if layer_cls is None:
            continue
        document.layers.append(layer_cls(entry["value"]))

    return image_path, document
