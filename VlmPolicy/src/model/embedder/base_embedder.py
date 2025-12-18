

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from model.embedder.data_structure.input import EncodingInput, EncodingParameters, OfflineEncodingInput
from model.embedder.data_structure.output import EncodingOutput, EncMetadata
import utils.file_io as IOUtils
import utils.directory as DirectoryUtils


import torch

class BaseEmbedder(ABC):

    def __init__(self):

        self._model_name: str = None

        self._global_cfg: Dict[str, Any] = None

        self._model_config_dir: str = None
        self._model_cfg: Dict[str, Any] = None

        self._local_path: str = None
        self._remote_port: int = None

        self._default_instruction = "Retrieve passage, table or image (maybe with a caption) that provides an answer to the given query."
        self._vector_dim = None

        return


    def _load_cfgs(self) -> None:
        """Initialize model configuration."""

        self._global_cfg = IOUtils.read_yaml(
            os.path.join(
                DirectoryUtils.get_embedder_config_dir(),
                "_base.yaml"
            )
        )

        self._model_config_dir = os.path.join(
            DirectoryUtils.get_embedder_config_dir(),
            self._model_name
        )
        self._model_cfg = IOUtils.read_yaml(os.path.join(self._model_config_dir, "_base.yaml"))


        self._local_path = os.path.join(
            self._global_cfg["models_path"],
            self._model_cfg["local_dirname"]
        )
        self._remote_port = self._model_cfg["remote_port"]
        
        self._vector_dim = self._model_cfg["vector_dim"]

        return


    @property
    def vector_dim(self) -> int:
        return self._vector_dim
    
    
    @abstractmethod
    def initiate(self) -> None:
        
        """
        Load the model.
        """
        raise NotImplementedError("initiate method not implemented.")
    
    
    
    @abstractmethod
    def free(self) -> None:
        
        """
        Delete the model.
        """
        raise NotImplementedError("free method not implemented.")
    
    
    
    @abstractmethod
    def encode_online(
        self,
        input: EncodingInput,
        parameters: EncodingParameters
    ) -> Tuple[
        EncodingOutput,
        EncMetadata
    ]:
        
        """
        Encode the queries.
        """
        raise NotImplementedError("encode method not implemented.")
    
    
    
    @abstractmethod
    def encode_list_online(
        self,
        input_list: List[EncodingInput],
        parameters: EncodingParameters
    ) -> Tuple[
        EncodingOutput,
        EncMetadata
    ]:
        
        """
        Encode the passages.
        """
        raise NotImplementedError("encode_list method not implemented.")
    
    
    
    @abstractmethod
    def encode_file_offline(
        self,
        file_info:  OfflineEncodingInput,
        parameters: EncodingParameters
    ) -> None:
        
        """
        Encode passages from a file, and save to disk.
        """
        raise NotImplementedError("encode_file method not implemented."
    )
    


    def _parse_file(
        self,
        file_path: str,
        parameters: EncodingParameters
    ) -> List[EncodingInput]:
        
        parsed = IOUtils.read_json_or_jsonl(file_path)
        
        encoding_inputs = []
        
        for item in parsed:
            id = item["id"]
            
            if "target" in item:
                item = item["target"]
            
            text = item["text"]
            image_paths = item["image_paths"]
            instruction = item.get("instruction", self._default_instruction)
        
            encoding_input = EncodingInput(
                id = id,
                instruction = instruction,
                text = text,
                image_paths = image_paths
            )
            
            encoding_inputs.append(encoding_input)
        
        return encoding_inputs
    
    def _get_slice_indices(
        self,
        total_items: int,
        world_size: int,
        rank: int
    ) -> Tuple[int, int]:
        """
        Calculate start and end indices for a uniform slice.

        Distributes items uniformly across ranks without losing any items.

        Args:
            total_items: Total number of items to distribute
            world_size: Number of parallel processes
            rank: Current process rank (0-indexed)

        Returns:
            Tuple of (start_idx, end_idx) for this rank's slice
        """

        # Calculate base size and remainder
        base_size = total_items // world_size
        remainder = total_items % world_size

        # Distribute remainder across first 'remainder' ranks
        # Each of the first 'remainder' ranks gets base_size + 1 items
        # Remaining ranks get base_size items
        if rank < remainder:
            start_idx = rank * (base_size + 1)
            end_idx = start_idx + base_size + 1
        else:
            start_idx = remainder * (base_size + 1) + (rank - remainder) * base_size
            end_idx = start_idx + base_size

        return start_idx, end_idx


    def _add_rank_suffix(
        self,
        filepath: str,
        world_size: int,
        rank: int
    ) -> str:
        """
        Add world_size and rank suffix to filename.

        Args:
            filepath: Original file path
            world_size: Number of parallel processes
            rank: Current process rank

        Returns:
            Modified filepath with suffix before extension
        """

        # Split into directory, filename, and extension
        directory = os.path.dirname(filepath)
        basename = os.path.basename(filepath)

        # Split basename into name and extension
        if '.' in basename:
            name, ext = basename.rsplit('.', 1)
            new_basename = f"{name}_world{world_size}_rank{rank}.{ext}"
        else:
            new_basename = f"{basename}_world{world_size}_rank{rank}"

        # Reconstruct path
        if directory:
            return os.path.join(directory, new_basename)
        else:
            return new_basename