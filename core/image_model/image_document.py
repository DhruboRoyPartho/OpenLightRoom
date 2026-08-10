import numpy as np
from core.processing.color_space import linear_to_display

# Layers are grouped into two fixed processing stages rather than applied in
# whatever order they were last edited (the earlier behavior): White
# Balance and Exposure are physically meaningful operations on scene-linear
# light, so they run first, in linear space; every other tone tool was
# designed and tuned to behave intuitively on a perceptually-encoded
# (display-referred) image, so they run after the linear -> display
# transform. This also makes rendering deterministic - the same set of
# edits always composes the same way, regardless of the order they were
# made in.
LINEAR_STAGE_ORDER = ["Temperature", "Tint", "Exposure"]
DISPLAY_STAGE_ORDER = [
    "Brightness", "Contrast", "Highlights", "Shadows", "Whites", "Blacks",
    "Parametric Curve", "Curve",
    "Vibrance", "Saturation", "Hue",
]


class ImageDocument:
    def __init__(self, base_image: np.ndarray):
        # float32, scene-linear, RGB, unclamped (highlight detail above
        # nominal white is preserved until the display transform, so it can
        # still be recovered by Exposure/Highlights). Never mutated after
        # construction - this is the one copy of the original data.
        self.base_image = base_image
        self.layers = []

        self.history = []
        self.redo_stack = []

        # Metadata carried alongside the pixels, for preservation on export.
        self.exif_data = {}
        self.icc_profile = None

    def add_layer(self, layer):
        self.layers.append(layer)
        self.history.append(f"{str(layer)}")
        self.redo_stack.clear()

    def render(self, geometry_override=None) -> np.ndarray:
        """geometry_override: a GeometryLayer to substitute for (or insert in
        place of) whatever "Crop" layer currently exists, for this render
        only - used for the crop tool's live preview while the user is still
        adjusting rotation/straighten-angle/crop before committing.

        Returns a float32 array in [0, 1], display-referred (gamma-encoded)
        - callers that need pixels for display or export convert from there.
        """
        image = self.base_image.copy()

        seen = set()
        deduped = []
        for layer in reversed(self.layers):
            name = str(layer)
            if name not in seen:
                seen.add(name)
                deduped.append(layer)
        deduped = list(reversed(deduped))

        by_name = {str(layer): layer for layer in deduped}
        if geometry_override is not None:
            by_name["Crop"] = geometry_override

        # 1. Geometry (crop/rotate/flip/straighten) - purely spatial, so it
        # doesn't matter which color stage it runs in; applied first so
        # everything downstream works on the final frame.
        crop_layer = by_name.get("Crop")
        if crop_layer is not None:
            image = crop_layer.apply(image)

        # 2. Linear-stage tools, on scene-linear light.
        for name in LINEAR_STAGE_ORDER:
            layer = by_name.get(name)
            if layer is not None:
                image = layer.apply(image)

        # 3. Linear -> display-referred transform.
        image = linear_to_display(image)

        # 4. Display-stage tools, on the perceptually-encoded image.
        for name in DISPLAY_STAGE_ORDER:
            layer = by_name.get(name)
            if layer is not None:
                image = layer.apply(image)

        return np.clip(image, 0.0, 1.0).astype(np.float32)

    def execute_command(self, command):
        command.execute()
        self.history.append(command)
        self.redo_stack.clear()

    def undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()
            self.redo_stack.append(command)

    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.history.append(command)
