

from typing import Dict, Type, Union

from model.mllm.base_mllm import BaseMLLM

from model.mllm.mllms.chatgpt5 import ChatGPT5, ChatGPT5_Factory
from model.mllm.mllms.qwen2_5_7b._base import Qwen2_5_7B, Qwen2_5_7B_Factory
from model.mllm.mllms.qwen2_5_vl_7b._base import Qwen2_5_VL_7B, Qwen2_5_VL_7B_Factory


MLLM_REGISTRY: Dict[str, Type[BaseMLLM]] = {
    "chatgpt5": ChatGPT5_Factory,
    "qwen2_5_7b": Qwen2_5_7B_Factory,
    "qwen2_5_vl_7b": Qwen2_5_VL_7B_Factory,
}