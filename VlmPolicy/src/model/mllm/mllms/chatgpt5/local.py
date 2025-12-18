from __future__ import annotations

from typing import List, Tuple

from model.mllm.mllms.chatgpt5._base import ChatGPT5
from model.mllm.data_structure.input import GenerationInput, GenerationParameters
from model.mllm.data_structure.output import GenerationOutput, GenMetadata


class ChatGPT5_Local(ChatGPT5):
    """
    Local implementation for ChatGPT-5.

    ChatGPT models cannot be run locally as they are cloud-based services.
    This class raises NotImplementedError for all methods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initiate(self) -> None:
        """Initialize the model and tokenizer."""
        raise NotImplementedError(
            "ChatGPT-5 cannot be run locally. "
            "It is a cloud-based service that requires API access. "
            "Use ChatGPT5_Client with LoadMode.REMOTE instead."
        )

    def infer(
        self,
        gen_input: GenerationInput,
        parameters: GenerationParameters = None,
    ) -> Tuple[GenerationOutput, GenMetadata]:
        """Generate a response based on the given prompt."""
        raise NotImplementedError(
            "ChatGPT-5 cannot be run locally. "
            "It is a cloud-based service that requires API access. "
            "Use ChatGPT5_Client with LoadMode.REMOTE instead."
        )

    def infer_list(
        self,
        gen_inputs: List[GenerationInput],
        parameters: GenerationParameters = None,
    ) -> Tuple[List[GenerationOutput], GenMetadata]:
        """Generate responses for a list of prompts."""
        raise NotImplementedError(
            "ChatGPT-5 cannot be run locally. "
            "It is a cloud-based service that requires API access. "
            "Use ChatGPT5_Client with LoadMode.REMOTE instead."
        )
