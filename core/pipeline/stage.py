from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


@dataclass
class Stage:
    """One named step of the render pipeline.

    A stage is exactly one of:
    - a set of adjustment layers (`layer_order`), applied in that fixed
      relative order for whichever of those layers actually exist on the
      document - this is what "White Balance/Exposure run in linear light,
      everything else runs after the display transform" turns into: two
      Stage instances with different layer_order lists, not an if/elif
      chain;
    - a pure image transform (`transform`, e.g. the linear -> display gamma
      encode), which doesn't correspond to any layer; or
    - a dynamic set of layers (`dynamic_prefix`), applying every layer
      whose name starts with that prefix, in document order - for tool
      types where the document can hold any number of instances at once
      (e.g. "Mask 1", "Mask 2", ...), unlike every other tool which is
      capped at one layer per fixed name.

    Exactly one of `layer_order` / `transform` / `dynamic_prefix` should be
    set.
    """

    name: str
    layer_order: Optional[list] = field(default=None)
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    dynamic_prefix: Optional[str] = None

    def __post_init__(self):
        set_count = sum(x is not None for x in (self.layer_order, self.transform, self.dynamic_prefix))
        if set_count != 1:
            raise ValueError(
                f"Stage {self.name!r} must set exactly one of layer_order, transform, or dynamic_prefix"
            )

    def apply(self, image: np.ndarray, by_name: dict, ordered_layers: Optional[list] = None) -> np.ndarray:
        if self.transform is not None:
            return self.transform(image)

        if self.dynamic_prefix is not None:
            layers = ordered_layers if ordered_layers is not None else list(by_name.values())
            for layer in layers:
                if str(layer).startswith(self.dynamic_prefix):
                    image = layer.apply(image)
            return image

        for layer_name in self.layer_order:
            layer = by_name.get(layer_name)
            if layer is not None:
                image = layer.apply(image)
        return image
