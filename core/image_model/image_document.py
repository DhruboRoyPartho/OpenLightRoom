import re
import numpy as np
from core.pipeline import build_default_pipeline
from core.processing.geometry import downscale_to_max_dimension

_MASK_NAME_RE = re.compile(r"Mask (\d+)")


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

        # Defines render order (which stage each named layer runs in, and
        # where the linear -> display transform sits between them) as data,
        # not logic in render() - see core/pipeline. Each document gets its
        # own instance so a future per-document layer registration can't
        # leak across documents.
        self.pipeline = build_default_pipeline()

    def add_layer(self, layer):
        self.layers.append(layer)
        self.history.append(f"{str(layer)}")
        self.redo_stack.clear()

    def next_mask_name(self) -> str:
        """The next unused "Mask N" pipeline name for a new
        MaskedAdjustmentLayer (see core/adjustment_layers/
        masked_adjustment_layer.py). Numbers are never reused, even after
        a mask is deleted - computed fresh from the current layer list
        rather than a persisted counter, so it self-heals correctly after
        a project reload with no extra serialized state."""
        highest = 0
        for layer in self.layers:
            match = _MASK_NAME_RE.fullmatch(str(layer))
            if match:
                highest = max(highest, int(match.group(1)))
        return f"Mask {highest + 1}"

    def render(self, geometry_override=None, max_dimension: int = None) -> np.ndarray:
        """geometry_override: a GeometryLayer to substitute for (or insert in
        place of) whatever "Crop" layer currently exists, for this render
        only - used for the crop tool's live preview while the user is still
        adjusting rotation/straighten-angle/crop before committing.

        max_dimension: if set, the working image is downscaled (before any
        pipeline stage, including Crop) so its longer side is at most this
        many pixels - a fast, high-quality (area-averaged) preview render
        for interactive editing; every per-pixel adjustment downstream is
        proportionally cheaper on a smaller image. None (the default)
        always renders at the source's full resolution, which is what the
        export/final-output path uses unconditionally - this parameter
        only ever affects on-screen preview rendering.

        Returns a float32 array in [0, 1], display-referred (gamma-encoded)
        - callers that need pixels for display or export convert from there.
        """
        image = downscale_to_max_dimension(self.base_image, max_dimension)
        image = image.copy()

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

        image = self.pipeline.render(image, by_name, deduped)

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
