"""
Directory utility functions for managing project paths.
"""

import os
from pathlib import Path


def get_project_root() -> str:
    """
    Get the root directory of the project.
    
    Returns:
        Absolute path to the project root directory
    """
    # Assuming this file is in src/utils/, go up 2 levels to get to project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return str(project_root)


def get_config_dir() -> str:
    """
    Get the configuration directory.
    
    Returns:
        Absolute path to the configs directory
    """
    return os.path.join(get_project_root(), "configs")


def get_mllm_config_dir() -> str:
    """
    Get the MLLM (Multimodal Large Language Model) configuration directory.
    
    Returns:
        Absolute path to the MLLM configs directory
    """
    return os.path.join(get_config_dir(), "mllm")


def get_embedder_config_dir() -> str:
    """
    Get the embedder configuration directory.
    
    Returns:
        Absolute path to the embedder configs directory
    """
    return os.path.join(get_config_dir(), "embedder")


def get_models_dir() -> str:
    """
    Get the models directory.
    
    Returns:
        Absolute path to the models directory
    """
    return os.path.join(get_project_root(), "models")


def ensure_dir_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)

