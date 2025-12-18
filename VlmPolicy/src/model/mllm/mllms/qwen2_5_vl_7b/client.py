from __future__ import annotations

import os
from typing import List, Tuple
from dataclasses import asdict
import requests

from model.mllm.mllms.qwen2_5_vl_7b._base import Qwen2_5_VL_7B
from model.mllm.data_structure.input import GenerationInput, GenerationParameters
from model.mllm.data_structure.output import GenerationOutput, GenMetadata
from model.mllm.data_structure.signature import (
    InferRequest,
    InferListRequest,
)
import utils.directory as DirectoryUtils
import utils.file_io as FileIOUtils





class Qwen2_5_VL_Client(Qwen2_5_VL_7B):
    """Client for Qwen-2.5-VL-7B remote inference server.

    This class provides the same API as Qwen2_5_VL_Local but sends requests
    to a remote server instead of running inference locally.
    """

    def __init__(self) -> None:
        """Initialize the client.

        Args:
            host: Server URL (e.g., "http://localhost:12345").
                  If None, will be loaded from config.
            timeout: Request timeout in seconds
        """
        super().__init__()

        # Load host from config if not provided
        config_dir = os.path.join(DirectoryUtils.get_mllm_config_dir(), "qwen2_5_vl_7b")
        config = FileIOUtils.read_yaml(os.path.join(config_dir, "_base.yaml"))

        client_host = config["client_host"]
        remote_port = config["remote_port"]

        host = f"http://{client_host}:{remote_port}"

        self._host = host.rstrip("/")
        self._timeout = config["timeout"]

        return


    def initiate(self) -> dict:
        """Initialize the remote model and return model information.

        Returns:
            dict: Model information including model_name, local_path, and status
        """
        url = f"{self._host}/initiate"
        try:
            response = requests.post(url, timeout=self._timeout)
            response.raise_for_status()
            model_info = response.json()

            # Store model info locally
            self._model_name = model_info.get("model_name")
            self._local_path = model_info.get("local_path")

            print()
            print("[Qwen2_5_VL_Client] Connected to remote server successfully.")
            print(f"Model: {self._model_name}")
            print()

            return model_info

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to initialize remote model: {e}")



    def infer(
        self,
        gen_input: GenerationInput,
        parameters: GenerationParameters = None,
    ) -> Tuple[GenerationOutput, GenMetadata]:
        """Single inference call via remote server.

        Args:
            gen_input: Input for generation
            parameters: Optional generation parameters

        Returns:
            Tuple of (GenerationOutput, GenMetadata)
        """
        url = f"{self._host}/infer"

        # Prepare request
        request_data = {
            "text_prompt":   gen_input.text_prompt,
            "system_prompt": gen_input.system_prompt,
            "image_paths":   gen_input.image_paths,
            "video_paths":   gen_input.video_paths,
        }

        if parameters is not None:
            request_data["parameters"] = asdict(parameters)

        try:
            response = requests.post(url, json=request_data, timeout=self._timeout)
            response.raise_for_status()
            result = response.json()

            # Parse response
            gen_output = GenerationOutput(text=result["text"])
            metadata = GenMetadata(**result["metadata"])

            return gen_output, metadata

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Inference request failed: {e}")



    def infer_list(
        self,
        gen_inputs: List[GenerationInput],
        parameters: GenerationParameters = None,
    ) -> Tuple[List[GenerationOutput], GenMetadata]:
        """Batch inference call via remote server.

        Args:
            gen_inputs: List of inputs for generation
            parameters: Optional generation parameters

        Returns:
            Tuple of (List[GenerationOutput], GenMetadata)
        """
        url = f"{self._host}/infer_list"

        # Prepare request
        inputs_data = []
        for gen_input in gen_inputs:
            input_dict = {
                "text_prompt":      gen_input.text_prompt,
                "system_prompt":    gen_input.system_prompt,
                "image_paths":      gen_input.image_paths,
                "video_paths":      gen_input.video_paths,
            }
            inputs_data.append(input_dict)

        request_data = {"inputs": inputs_data}

        if parameters is not None:
            request_data["parameters"] = asdict(parameters)

        try:
            response = requests.post(url, json = request_data, timeout = self._timeout)
            response.raise_for_status()
            result = response.json()

            # Parse response
            gen_outputs = [GenerationOutput(text=text) for text in result["outputs"]]
            metadata = GenMetadata(**result["metadata"])

            return gen_outputs, metadata

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Batch inference request failed: {e}")


if __name__ == "__main__":

    # Example usage - will automatically load host from config
    client = Qwen2_5_VL_Client()

    # Initialize remote model
    model_info = client.initiate()
    print(f"Model info: {model_info}")

    # Test with cat image
    cat_image_path = "/root/omdr_workspace/src/model/test_files/cat.png"

    # Single inference with image
    input1 = GenerationInput(
        text_prompt="What do you see in this image? Describe it in detail.",
        image_paths=[cat_image_path]
    )
    output1, metadata1 = client.infer(input1)
    print(f"\n===== Single Inference (with image) =====\n{output1.text}\n")
    print(f"Metadata: {metadata1}\n")

    print()
    print()
    print()

    # Batch inference
    inputs = [
        GenerationInput(
            text_prompt = "What is the capital of France?"
        ),
        GenerationInput(
            text_prompt = "What animal is this?",
            image_paths = [cat_image_path]
        ),
        GenerationInput(
            text_prompt = "What is the capital of Italy?"
        ),
    ]

    outputs, metadata = client.infer_list(inputs)

    for i, out in enumerate(outputs, 1):
        print(f"\n===== Output {i} =====\n{out.text}\n")
    print(f"\n===== Metadata =====\n{metadata}\n")

    pass
