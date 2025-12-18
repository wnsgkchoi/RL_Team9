

from __future__ import annotations
from dataclasses import dataclass
import json

from typing import List, Optional, Union, Tuple, Dict, Any

import torch


TIME_MULTIPLIER_MS = 1000.0








@dataclass
class EncodingOutput:
    
    tensors:        torch.Tensor = None
    index  :        Dict[str, Any] = None


    





@dataclass
class EncMetadata:

    time:               float = None

    time_preprocess:    float = None
    time_infer:         float = None
    time_postprocess:   float = None

    num_inputs:         int   = None
    batch_size:         int   = None
    batch_iterations:   int   = None


    def __str__(self) -> str:
        # Convert to dict, handling nested dataclass
        data = {}
        for key, value in self.__dict__.items():
            if value is None:
                data[key] = None
            else:
                data[key] = value
        return json.dumps(data, indent = 4)

    def __iadd__(self, other: EncMetadata) -> EncMetadata:
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

        return self


