

from __future__ import annotations

from typing import Type, Dict
from model.embedder.base_embedder import BaseEmbedder
from model.mllm.utils.constants import LoadMode


class NVEmbedV2(BaseEmbedder):
    """
    Base class for NV-Embed-v2 model implementations.
    Concrete implementations should be created via the factory.

    Note: NV-Embed-v2 is a text-only embedder. Image inputs will trigger warnings.
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._model_name = "nvembed_v2"
        self._load_cfgs()

        return


    def _warn_multimodal_inputs(self, enc_input) -> None:
        """Warn if image inputs are provided for this text-only model."""

        if enc_input.image_paths and len(enc_input.image_paths) > 0:
            print(f"[Warning][{self._model_name}] NV-Embed-v2 is a text-only embedder. Image inputs will be ignored.")

        return


class NVEmbedV2_Factory:
    """
    Factory class for creating NV-Embed-v2 model instances based on load mode.

    Usage:
        factory = NVEmbedV2Factory()
        model = factory.create(mode=LoadMode.LOCAL, **kwargs)
    """

    # Class-level registry shared across factory instances
    _registry: Dict[LoadMode, Type[NVEmbedV2]] = {}

    def __init__(self) -> None:
        # Instance initializer retained for potential future state
        self._instance_name: str = "NVEmbedV2Factory"

    @classmethod
    def register(cls, mode: LoadMode, implementation: Type[NVEmbedV2]) -> None:
        """
        Register a concrete implementation for a specific load mode.

        Args:
            mode: The LoadMode (LOCAL or REMOTE)
            implementation: The concrete class to instantiate for this mode
        """
        cls._registry[mode] = implementation

    def create(self, mode: LoadMode, **kwargs) -> NVEmbedV2:
        """
        Create an NV-Embed-v2 instance based on the specified load mode.

        Args:
            mode: LoadMode.LOCAL for local inference or LoadMode.REMOTE for remote server
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            An instance of the appropriate NV-Embed-v2 implementation

        Raises:
            ValueError: If the mode is not registered
        """
        if mode not in self._registry:
            raise ValueError(
                f"Unknown load mode: {mode}. "
                f"Available modes: {list(self._registry.keys())}"
            )

        implementation_class = self._registry[mode]
        print(f"Creating {implementation_class.__name__} for mode: {mode.value}")
        return implementation_class(**kwargs)
