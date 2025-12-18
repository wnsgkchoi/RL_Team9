"""
File I/O utility functions for reading and writing various file formats.
"""

import json
import os
from typing import Any, Dict, List, Union
import yaml


def read_yaml(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing the YAML contents
        
    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If the file cannot be parsed as YAML
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Write a dictionary to a YAML file.
    
    Args:
        data: Dictionary to write
        file_path: Path to the output YAML file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def read_json(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Read a JSON file and return its contents.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary or list containing the JSON contents
        
    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file cannot be parsed as JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL (JSON Lines) file and return its contents as a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
        
    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If a line cannot be parsed as JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def read_json_or_jsonl(file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Read a JSON or JSONL file automatically based on extension or content.
    
    Args:
        file_path: Path to the JSON/JSONL file
        
    Returns:
        Dictionary (for JSON) or list of dictionaries (for JSONL)
        
    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file cannot be parsed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try to determine format from extension
    if file_path.endswith('.jsonl'):
        return read_jsonl(file_path)
    elif file_path.endswith('.json'):
        return read_json(file_path)
    
    # If extension is ambiguous, try parsing as JSON first, then JSONL
    try:
        return read_json(file_path)
    except json.JSONDecodeError:
        return read_jsonl(file_path)


def write_json_file(data: Union[Dict[str, Any], List[Any]], file_path: str, indent: int = 2) -> None:
    """
    Write data to a JSON file.
    
    Args:
        data: Dictionary or list to write
        file_path: Path to the output JSON file
        indent: Number of spaces for indentation (default: 2)
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def write_jsonl_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Write a list of dictionaries to a JSONL file.
    
    Args:
        data: List of dictionaries to write
        file_path: Path to the output JSONL file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def read_text_file(file_path: str) -> str:
    """
    Read a text file and return its contents as a string.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        String containing the file contents
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(text: str, file_path: str) -> None:
    """
    Write a string to a text file.
    
    Args:
        text: String to write
        file_path: Path to the output text file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

