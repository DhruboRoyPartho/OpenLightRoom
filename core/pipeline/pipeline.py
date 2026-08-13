import numpy as np

from core.pipeline.stage import Stage


class Pipeline:
    """An ordered, registrable list of Stages.

    Deterministic: the same set of edits always renders the same way
    regardless of the order the user made them in, because that order lives
    here, not in edit history. New tools extend an existing Pipeline
    instance via register_layer() (or add a whole new Stage via add_stage())
    instead of editing this class or ImageDocument.render().
    """

    def __init__(self, stages=None):
        self._stages: list[Stage] = list(stages) if stages else []

    def add_stage(self, stage: Stage, index: int = None) -> None:
        if index is None:
            self._stages.append(stage)
        else:
            self._stages.insert(index, stage)

    def stage(self, name: str) -> Stage:
        for s in self._stages:
            if s.name == name:
                return s
        raise KeyError(f"No stage named {name!r} in pipeline")

    def register_layer(self, stage_name: str, layer_name: str, position: int = None) -> None:
        """Add a new adjustment-layer name into an existing stage's
        layer_order (idempotent - registering the same name twice is a
        no-op), so a new tool can extend the pipeline without editing this
        module."""
        st = self.stage(stage_name)
        if st.layer_order is None:
            raise ValueError(f"Stage {stage_name!r} is a transform stage, not a layer stage")
        if layer_name in st.layer_order:
            return
        if position is None:
            st.layer_order.append(layer_name)
        else:
            st.layer_order.insert(position, layer_name)

    def render(self, base_image: np.ndarray, by_name: dict, ordered_layers: list = None) -> np.ndarray:
        """ordered_layers: the document's deduped layers in their original
        (creation/list) order - needed by a dynamic_prefix Stage, which
        can't be driven by by_name alone since it doesn't know its
        matching layers' names in advance. Optional and defaults to
        by_name's own insertion order (which already matches, since
        by_name is built by iterating the same deduped list) - existing
        callers that never had a dynamic_prefix stage don't need to pass
        it."""
        image = base_image
        for st in self._stages:
            image = st.apply(image, by_name, ordered_layers)
        return image
