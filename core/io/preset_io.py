import json
import os
import re
import sys

from core.io.project_io import serialize_layers, deserialize_layers

# Presets live in a plain folder of JSON files (not inside any specific
# .olrproj project), so they're reusable across every image/project - this
# is the "reusable, versionable look" model, distinct from a project file's
# "this specific image's edit state".
PRESET_VERSION = 1
PRESET_EXTENSION = ".json"


def _default_presets_dir() -> str:
    """Presets are user data the app writes to at runtime (unlike
    assets/, which is read-only bundled content - see
    interface/gui/assets.py), so once installed/frozen they can't live
    next to the executable: a standard installer puts that under
    Program Files, which a non-admin user can't write to (this is
    exactly the PermissionError a packaged build hit before this
    existed). Frozen builds use the per-user app-data directory instead;
    running from source keeps using the repo's own presets/ folder, so
    existing dev/test presets and expectations don't change."""
    if getattr(sys, "frozen", False):
        base = os.environ.get("APPDATA") or os.path.expanduser("~/.config")
        return os.path.join(base, "OpenLightRoom", "presets")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "presets")


PRESETS_DIR = _default_presets_dir()

# Geometry (crop/rotate/flip/straighten) describes how one specific photo
# was framed, not a reusable color/tone "look" - Lightroom-style presets
# exclude it by the same reasoning, so it's filtered out of both what gets
# saved into a preset and what gets applied from one. Local (masked)
# adjustments are excluded for the same reason: a mask's geometry (a
# brush stroke, a radial placed over one specific subject) is tied to one
# photo's composition, not a portable "look" - real masking panels
# exclude local adjustments from standard presets too.
_EXCLUDED_FROM_PRESETS = {"Crop"}


def _is_excluded_from_presets(layer_name: str) -> bool:
    return layer_name in _EXCLUDED_FROM_PRESETS or layer_name.startswith("Mask ")


def _ensure_presets_dir():
    os.makedirs(PRESETS_DIR, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    if not safe:
        raise ValueError("Preset name can't be empty.")
    return safe


def _path_for(name: str) -> str:
    return os.path.join(PRESETS_DIR, _sanitize_filename(name) + PRESET_EXTENSION)


def list_presets() -> list:
    """Returns preset names (not file paths), sorted case-insensitively."""
    _ensure_presets_dir()
    names = []
    for filename in os.listdir(PRESETS_DIR):
        if filename.lower().endswith(PRESET_EXTENSION):
            names.append(filename[: -len(PRESET_EXTENSION)])
    return sorted(names, key=str.lower)


def preset_exists(name: str) -> bool:
    return os.path.isfile(_path_for(name))


def layers_for_preset(layers) -> list:
    """Filters out layers that shouldn't be captured in a reusable preset
    (Crop, and any "Mask N" local adjustment - see
    _EXCLUDED_FROM_PRESETS/_is_excluded_from_presets)."""
    return [l for l in layers if not _is_excluded_from_presets(str(l))]


def save_preset(name: str, layers, overwrite: bool = False) -> str:
    """layers: the adjustment layers to capture (geometry is filtered out
    automatically). Returns the path written. Raises FileExistsError if a
    preset with this name already exists and overwrite is False."""
    _ensure_presets_dir()
    path = _path_for(name)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"A preset named '{name}' already exists.")

    data = {
        "version": PRESET_VERSION,
        "name": name,
        "layers": serialize_layers(layers_for_preset(layers)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_preset(name: str) -> list:
    """Returns the list of adjustment layer objects stored in the named
    preset. Raises FileNotFoundError if it doesn't exist."""
    path = _path_for(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No preset named '{name}'.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return deserialize_layers(data.get("layers", []))


def delete_preset(name: str):
    path = _path_for(name)
    if os.path.isfile(path):
        os.remove(path)


def duplicate_preset(name: str, new_name: str) -> str:
    """Returns the path of the new copy. Raises FileNotFoundError if the
    source doesn't exist, FileExistsError if new_name is already taken."""
    layers = load_preset(name)  # raises FileNotFoundError if missing
    return save_preset(new_name, layers, overwrite=False)


def export_preset(name: str, dest_path: str):
    """Copies a preset's JSON out to an arbitrary file path, for sharing."""
    path = _path_for(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No preset named '{name}'.")
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(data)


def import_preset(src_path: str, name: str = None, overwrite: bool = False) -> str:
    """Reads an external preset JSON file and adds it to the presets
    folder under `name` (or the name recorded inside the file, or the
    source filename, in that preference order). Returns the preset name
    actually used."""
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preset_name = name or data.get("name") or os.path.splitext(os.path.basename(src_path))[0]
    layers = deserialize_layers(data.get("layers", []))
    save_preset(preset_name, layers, overwrite=overwrite)
    return preset_name
