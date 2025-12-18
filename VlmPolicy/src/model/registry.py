
from typing import Dict, Type, Union

from model.mllm.registry import MLLM_REGISTRY
from model.mllm.base_mllm import BaseMLLM

from model.embedder.registry import EMBEDDER_REGISTRY
from model.embedder.base_embedder import BaseEmbedder

MODEL_REGISTRY = MLLM_REGISTRY | EMBEDDER_REGISTRY
