
# Import abstractclass and abstract method

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

from model.mllm.data_structure.input  import GenerationInput, GenerationParameters
from model.mllm.data_structure.output import GenerationOutput, GenMetadata

import utils.directory as DirectoryUtils
import utils.file_io as FileIOUtils


class BaseMLLM(ABC):
    
    def __init__(self):
        
        self._model_name: str = None
        
        self._global_cfg: Dict[str, Any] = None
        
        self._model_config_dir: str = None
        self._model_cfg: Dict[str, Any] = None
        
        self._local_path: str = None
        self._remote_port: int = None
        
        self._tokenizer: Any = None
        self._model: Any = None
    
        
        pass
    
    
    def _load_cfgs(self) -> None:
        """Initialize model configuration."""
        
        self._global_cfg = FileIOUtils.read_yaml(
            os.path.join(
                DirectoryUtils.get_mllm_config_dir(),
                "_base.yaml"
            )
        )
        
        self._model_config_dir = os.path.join(
            DirectoryUtils.get_mllm_config_dir(),
            self._model_name
        )
        self._model_cfg = FileIOUtils.read_yaml(os.path.join(self._model_config_dir, "_base.yaml"))
        

        self._local_path = os.path.join(
            self._global_cfg["models_path"],
            self._model_cfg["local_dirname"]
        )
        self._remote_port = self._model_cfg["remote_port"]
        
        return
    
    
    def _warn_multimodal_inputs(
        self,
        gen_input: GenerationInput
    ) -> None:
        """Warn if multimodal inputs are provided but not supported."""
        
        if gen_input.image_paths and len(gen_input.image_paths) > 0:
            print(f"[Warning][{self._model_name}] This model does not support image input.")
                  
        if gen_input.video_paths and len(gen_input.video_paths) > 0:
            print(f"[Warning][{self._model_name}] This model does not support video input.")
            
        return
    
    
    @abstractmethod
    def initiate(self) -> None:
        """Initialize the model and tokenizer."""
        
        raise NotImplementedError("Model initiation not implemented yet.")
    
    
    @abstractmethod
    def infer(
        self,
        gen_input: GenerationInput,
        parameters: GenerationParameters,
        **kwargs: Any
    ) -> Tuple[
            GenerationOutput,
            GenMetadata
    ]:
        """Generate a response based on the given prompt."""
        
        raise NotImplementedError("Inference not implemented yet.")
    
    
    @abstractmethod
    def infer_list(
        self,
        gen_inputs: List[GenerationInput],
        parameters: GenerationParameters,
        **kwargs: Any
    ) -> Tuple[
        List[GenerationOutput],
        GenMetadata
    ]:
        """Generate responses for a list of prompts."""
        
        raise NotImplementedError("Batch inference not implemented yet.")
    
    