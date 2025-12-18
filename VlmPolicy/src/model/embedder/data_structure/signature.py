from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


from model.embedder.data_structure.input import EncodingInput, EncodingParameters


class EncodeRequest(BaseModel):
    """Request model for single encoding.

    Attributes:
        id: Input ID
        instruction: Optional instruction text
        text: Text to encode
        image_paths: Optional list of image paths
        parameters: Optional encoding parameters as dict
    """
    id:             str
    instruction:    Optional[str] = None
    text:           Optional[str] = None
    image_paths:    Optional[List[str]] = None
    parameters:     Optional[Dict[str, Any]] = None


class EncodeListRequest(BaseModel):
    """Request model for batch encoding.

    Attributes:
        inputs: List of input dictionaries, each containing id, text, etc.
        parameters: Optional encoding parameters as dict
    """
    inputs:         List[Dict[str, Any]]
    parameters:     Optional[Dict[str, Any]] = None


class EncodeResponse(BaseModel):
    """Response model for single encoding.

    Attributes:
        embeddings: The embedding tensor as a list
        metadata: Encoding metadata as dict
    """
    embeddings:     List[List[float]]
    metadata:       Dict[str, Any]


class EncodeListResponse(BaseModel):
    """Response model for batch encoding.

    Attributes:
        embeddings: List of embedding tensors
        index: Mapping from IDs to embedding indices
        metadata: Encoding metadata as dict
    """
    embeddings:     List[List[float]]
    index:          Dict[str, int]
    metadata:       Dict[str, Any]


class InitiateResponse(BaseModel):
    """Response model for model initialization info.

    Attributes:
        model_name: Name of the model
        local_path: Path to the model
        vector_dim: Dimension of embeddings
        status: Initialization status
    """
    model_name:     str
    local_path:     str
    vector_dim:     int
    status:         str
