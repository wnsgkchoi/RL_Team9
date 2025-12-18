from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict, Any


from model.embedder.data_structure.output import EncMetadata




@dataclass
class EncodingInput:
    
    id:             Union[str, List]
    
    instruction:    str = None
    
    text:           str = None
    image_paths:    List[str] = None
    




@dataclass
class OfflineEncodingInput:
    
    in_filepath:            str = None
    
    out_embeddings_path:    str = None
    out_index_path:         str = None
    out_metadata_path:      EncMetadata = None
    
    def sanity_check(self) -> None:
        # Check if in_filepath is .json or .jsonl
        if not (self.in_filepath.endswith(".json") or self.in_filepath.endswith(".jsonl")):
            raise ValueError("in_filepath must be a .json or .jsonl file.")
        # Check if out_embeddings_path ends with .pt
        if not self.out_embeddings_path.endswith(".pt"):
            raise ValueError("out_embeddings_path must end with .pt")
        # Check if out_index_path ends with .json
        if not self.out_index_path.endswith(".json"):
            raise ValueError("out_index_path must end with .json")
        return






@dataclass
class EncodingParameters:

    show_progress:      bool            = None

    batch_size:         int             = None
    max_length:         int             = None

    world_size:         int             = None
    rank:               int             = None

    write_file:         bool            = None
    filepath:           str             = None
    
    
    def apply_default_settings(
        self,
        default_params: EncodingParameters
    ) -> None:

        """Apply default settings for any parameters that are None."""

        if self.show_progress is None:
            self.show_progress = default_params.show_progress

        if self.batch_size is None:
            self.batch_size = default_params.batch_size

        if self.max_length is None:
            self.max_length = default_params.max_length

        if self.world_size is None:
            self.world_size = default_params.world_size

        if self.rank is None:
            self.rank = default_params.rank

        if self.write_file is None:
            self.write_file = default_params.write_file

        if self.filepath is None:
            self.filepath = default_params.filepath

        return


    def __iadd__(
        self,
        default_params: EncodingParameters
    ) -> EncodingParameters:
        
        """Override += operator to apply default settings.

        Usage: params += default_params
        """
        self.apply_default_settings(default_params)
        return self
    