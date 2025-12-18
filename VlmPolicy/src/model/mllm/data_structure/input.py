from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict, Any


@dataclass
class GenerationInput:
    text_prompt:    str = None
    system_prompt:  Optional[str] = None
    image_paths:    List[str] = None
    video_paths:    List[str] = None
    text_map:       str = None

    # def __post_init__(self):
    #     # Convert None to empty lists for image_paths and video_paths
    #     if self.image_paths is None:
    #         self.image_paths = []
    #     if self.video_paths is None:
    #         self.video_paths = []



@dataclass
class APIParameters:
    effort: str = "low"


@dataclass
class GenerationParameters:

    show_progress:      bool = None
    
    batch_size:         int = None
    
    max_tokens:         int = None
    
    do_sample:          bool = None
    
    top_p:              float = None
    temperature:        float = None
    repetition_penalty: float = None
    
    api_parameters:    APIParameters = None
    
    
    def apply_default_settings(
        self,
        default_params: GenerationParameters) -> None:
        """Apply default settings for any parameters that are None."""

        if self.show_progress is None:
            self.show_progress = default_params.show_progress

        if self.batch_size is None:
            self.batch_size = default_params.batch_size

        if self.max_tokens is None:
            self.max_tokens = default_params.max_tokens

        if self.do_sample is None:
            self.do_sample = default_params.do_sample

        if self.top_p is None:
            self.top_p = default_params.top_p

        if self.temperature is None:
            self.temperature = default_params.temperature

        if self.repetition_penalty is None:
            self.repetition_penalty = default_params.repetition_penalty

        return

    def __iadd__(self, default_params: GenerationParameters) -> GenerationParameters:
        """Override += operator to apply default settings.

        Usage: params += default_params
        """
        self.apply_default_settings(default_params)
        return self


