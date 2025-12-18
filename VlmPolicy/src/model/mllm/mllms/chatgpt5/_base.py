

from __future__ import annotations

from typing import Type, Dict
from model.mllm.base_mllm import BaseMLLM
from model.mllm.utils.constants import LoadMode


class ChatGPT5(BaseMLLM):
    """
    Base class for ChatGPT-5 model implementations.
    Concrete implementations should be created via the factory.
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._model_name = "chatgpt5"
        self._load_cfgs()

        return


class ChatGPT5_Factory:
    """
    Factory class for creating ChatGPT-5 model instances based on load mode.

    Usage:
        factory = ChatGPT5_Factory()
        model = factory.create(mode=LoadMode.REMOTE, **kwargs)
    """

    # Class-level registry shared across factory instances
    _registry: Dict[LoadMode, Type[ChatGPT5]] = {}

    def __init__(self) -> None:
        # Instance initializer retained for potential future state
        self._instance_name: str = "ChatGPT5_Factory"

    @classmethod
    def register(cls, mode: LoadMode, implementation: Type[ChatGPT5]) -> None:
        """
        Register a concrete implementation for a specific load mode.

        Args:
            mode: The LoadMode (LOCAL or REMOTE)
            implementation: The concrete class to instantiate for this mode
        """
        cls._registry[mode] = implementation

    def create(self, mode: LoadMode, **kwargs) -> ChatGPT5:
        """
        Create a ChatGPT-5 instance based on the specified load mode.

        Args:
            mode: LoadMode.LOCAL for local inference or LoadMode.REMOTE for remote server
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            An instance of the appropriate ChatGPT5 implementation

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
