

from __future__ import annotations
from dataclasses import dataclass
import json

from typing import List, Optional, Union, Tuple, Dict, Any


TIME_MULTIPLIER_MS = 1000.0




@dataclass
class APIUsage:
    
    model:                  str = ""
    rate_info:              Dict[str, float] = None
    
    input_tokens:           int = 0
    output_tokens:          int = 0
    cached_input_tokens:    int = 0
    
    estimated_cost_usd:     float = 0.0

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent = 4)
    
    def __iadd__(self, other: APIUsage) -> APIUsage:
        """In-place addition to aggregate API usage from multiple generations."""
        
        # Skip model
        # Skip rate_info
        
        if other.input_tokens is not None:
            self.input_tokens += other.input_tokens
        
        if other.output_tokens is not None:
            self.output_tokens += other.output_tokens
        
        if other.cached_input_tokens is not None:
            self.cached_input_tokens += other.cached_input_tokens
        
        if other.estimated_cost_usd is not None:
            self.estimated_cost_usd += other.estimated_cost_usd
        
        return self



@dataclass
class GenMetadata:
    
    time:               float = None
    
    time_preprocess:    float = None
    time_infer:         float = None
    time_postprocess:   float = None
    
    num_inputs:         int   = None
    batch_size:         int   = None
    batch_iterations:   int   = None
    
    api_usage:          APIUsage = None

    def __str__(self) -> str:
        # Convert to dict, handling nested dataclass
        data = {}
        for key, value in self.__dict__.items():
            if value is None:
                data[key] = None
            elif isinstance(value, APIUsage):
                data[key] = value.__dict__
            else:
                data[key] = value
        return json.dumps(data, indent = 4)

    def __iadd__(self, other: GenMetadata) -> GenMetadata:
        """In-place addition to aggregate metadata from multiple generations."""
        
        if other.time is not None:
            if self.time is None:
                self.time = 0.0
            self.time += other.time
        
        if other.time_preprocess is not None:
            if self.time_preprocess is None:
                self.time_preprocess = 0.0
            self.time_preprocess += other.time_preprocess
        
        if other.time_infer is not None:
            if self.time_infer is None:
                self.time_infer = 0.0
            self.time_infer += other.time_infer
        
        if other.time_postprocess is not None:
            if self.time_postprocess is None:
                self.time_postprocess = 0.0
            self.time_postprocess += other.time_postprocess
        
        # Skip num_inputs
        # Skip batch_size
        # Skip batch_iterations

        if other.api_usage is not None:
            if self.api_usage is None:
                self.api_usage = APIUsage()
            self.api_usage += other.api_usage
        
        return self
    
    


@dataclass
class GenerationOutput:
    
    text: str
    