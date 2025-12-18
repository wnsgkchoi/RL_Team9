from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


from model.mllm.data_structure.input import GenerationInput, GenerationParameters   


class InferRequest(BaseModel):
    """Request model for single inference.

    Attributes:
        text_prompt: The text prompt for generation
        system_prompt: Optional system prompt
        image_paths: Optional list of image paths
        video_paths: Optional list of video paths
        parameters: Optional generation parameters as dict
    """
    text_prompt:    str
    system_prompt:  Optional[str] = None
    image_paths:    Optional[List[str]] = None
    video_paths:    Optional[List[str]] = None
    parameters:     Optional[Dict[str, Any]] = None


class InferListRequest(BaseModel):
    """Request model for batch inference.

    Attributes:
        inputs: List of input dictionaries, each containing text_prompt, etc.
        parameters: Optional generation parameters as dict
    """
    inputs:         List[Dict[str, Any]]
    parameters:     Optional[Dict[str, Any]] = None


class InferResponse(BaseModel):
    """Response model for single inference.

    Attributes:
        text: Generated text
        metadata: Generation metadata as dict
    """
    text:           str
    metadata:       Dict[str, Any]


class InferListResponse(BaseModel):
    """Response model for batch inference.

    Attributes:
        outputs: List of generated texts
        metadata: Generation metadata as dict
    """
    outputs:        List[str]
    metadata:       Dict[str, Any]


class InitiateResponse(BaseModel):
    """Response model for model initialization info.

    Attributes:
        model_name: Name of the model
        local_path: Path to the model
        status: Initialization status
    """
    model_name:     str
    local_path:     str
    status:         str
