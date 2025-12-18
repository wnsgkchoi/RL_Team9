
from typing import Dict, Type, Union

from model.embedder.base_embedder import BaseEmbedder

from model.embedder.embedders.mmembed._base import MMEmbed, MMEmbed_Factory
from model.embedder.embedders.nvembed_v2._base import NVEmbedV2, NVEmbedV2_Factory


EMBEDDER_REGISTRY: Dict[str, Type[BaseEmbedder]] = {
    "mmembed": MMEmbed_Factory,
    "nvembed_v2": NVEmbedV2_Factory,
}