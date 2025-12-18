from __future__ import annotations

import os
import tempfile
from typing import List, Tuple
from dataclasses import asdict
import requests
import torch

from model.embedder.embedders.mmembed._base import MMEmbed
from model.embedder.data_structure.input import EncodingInput, EncodingParameters, OfflineEncodingInput
from model.embedder.data_structure.output import EncodingOutput, EncMetadata
from model.embedder.data_structure.signature import (
    EncodeRequest,
    EncodeListRequest,
)
import utils.directory as DirectoryUtils
import utils.file_io as FileIOUtils
import utils.file_io as IOUtils





class MMEmbedClient(MMEmbed):
    """Client for MM-Embed remote encoding server.

    This class provides the same API as MMEmbedLocal but sends requests
    to a remote server instead of running encoding locally.
    """

    def __init__(self) -> None:
        """Initialize the client.

        Args:
            host: Server URL (e.g., "http://localhost:13456").
                  If None, will be loaded from config.
            timeout: Request timeout in seconds
        """
        super().__init__()

        # Load host from config if not provided
        config_dir = os.path.join(DirectoryUtils.get_embedder_config_dir(), "mmembed")
        config = FileIOUtils.read_yaml(os.path.join(config_dir, "_base.yaml"))

        client_host = config["client_host"]
        remote_port = config["remote_port"]

        host = f"http://{client_host}:{remote_port}"

        self._host = host.rstrip("/")
        self._timeout = config["timeout"]

        # Default parameters for file operations
        self._default_parameters = EncodingParameters(
            show_progress   = True,
            batch_size      = 4,
            max_length      = 256,
            world_size      = 1,
            rank            = 0
        )

        return


    def initiate(self) -> dict:
        """Initialize the remote model and return model information.

        Returns:
            dict: Model information including model_name, local_path, vector_dim, and status
        """
        url = f"{self._host}/initiate"
        try:
            response = requests.post(url, timeout=self._timeout)
            response.raise_for_status()
            model_info = response.json()

            # Store model info locally
            self._model_name = model_info.get("model_name")
            self._local_path = model_info.get("local_path")
            self._vector_dim = model_info.get("vector_dim")

            print()
            print("[MMEmbedClient] Connected to remote server successfully.")
            print(f"Model: {self._model_name}")
            print(f"Vector dimension: {self._vector_dim}")
            print()

            return model_info

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to initialize remote model: {e}")


    def free(self) -> None:
        """No-op for remote client. Server manages model lifecycle."""
        print()
        print("[MMEmbedClient] Model lifecycle managed by remote server.")
        print()
        return


    def encode_online(
        self,
        input: EncodingInput,
        parameters: EncodingParameters = None
    ) -> Tuple[
        EncodingOutput, 
        EncMetadata
    ]:
        
        """Single encoding call via remote server.

        Args:
            input: Input for encoding
            parameters: Optional encoding parameters

        Returns:
            Tuple of (EncodingOutput, EncMetadata)
        """
        url = f"{self._host}/encode_online"

        # Create temporary file for tensor transfer
        temp_fd, temp_filepath = tempfile.mkstemp(suffix=".pt")
        os.close(temp_fd)  # Close the file descriptor

        try:
            # Set file writing parameters
            if parameters is None:
                parameters = EncodingParameters()
            parameters.write_file = True
            parameters.filepath = temp_filepath

            # Prepare request
            request_data = {
                "id":           input.id,
                "instruction":  input.instruction,
                "text":         input.text,
                "image_paths":  input.image_paths,
            }

            if parameters is not None:
                request_data["parameters"] = asdict(parameters)

            response = requests.post(url, json=request_data, timeout=self._timeout)
            response.raise_for_status()
            result = response.json()

            # Load tensor from file instead of from response
            embeddings_tensor = torch.load(temp_filepath)

            # Create index (single item)
            index = {str(input.id): 0}

            enc_output = EncodingOutput(
                tensors=embeddings_tensor,
                index=index
            )
            metadata = EncMetadata(**result["metadata"])

            return enc_output, metadata

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Encoding request failed: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)



    def encode_list_online(
        self,
        input_list: List[EncodingInput],
        parameters: EncodingParameters = None
    ) -> Tuple[
        EncodingOutput, 
        EncMetadata
    ]:
        
        """Batch encoding call via remote server.

        Args:
            input_list: List of inputs for encoding
            parameters: Optional encoding parameters

        Returns:
            Tuple of (EncodingOutput, EncMetadata)
        """
        url = f"{self._host}/encode_list_online"

        # Create temporary file for tensor transfer
        temp_fd, temp_filepath = tempfile.mkstemp(suffix=".pt")
        os.close(temp_fd)  # Close the file descriptor

        try:
            # Set file writing parameters
            if parameters is None:
                parameters = EncodingParameters()
            parameters.write_file = True
            parameters.filepath = temp_filepath

            # Prepare request
            inputs_data = []
            for enc_input in input_list:
                input_dict = {
                    "id":           enc_input.id,
                    "instruction":  enc_input.instruction,
                    "text":         enc_input.text,
                    "image_paths":  enc_input.image_paths,
                }
                inputs_data.append(input_dict)

            request_data = {"inputs": inputs_data}

            if parameters is not None:
                request_data["parameters"] = asdict(parameters)

            response = requests.post(url, json = request_data, timeout = self._timeout)
            response.raise_for_status()
            result = response.json()

            # Load tensor from file instead of from response
            embeddings_tensor = torch.load(temp_filepath)

            enc_output = EncodingOutput(
                tensors = embeddings_tensor,
                index = result["index"]
            )
            metadata = EncMetadata(**result["metadata"])

            return enc_output, metadata

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Batch encoding request failed: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)


    def encode_file_offline(
        self,
        file_info: OfflineEncodingInput,
        parameters: EncodingParameters = None
    ) -> None:
        """
        Encode passages from a file and save to disk.

        Delegates to encode_list_online for remote processing.
        """

        # Sanity check
        file_info.sanity_check()

        # Apply default parameters to get world_size and rank
        if not parameters:
            parameters = EncodingParameters()
        parameters += self._default_parameters

        world_size = parameters.world_size
        rank = parameters.rank

        # Parse the input file
        encoding_inputs = self._parse_file(
            file_info.in_filepath,
            parameters
        )

        # Encode the inputs via remote server (slicing handled on client side for file operations)
        # Apply distributed slicing on client side for file operations
        total_items = len(encoding_inputs)
        if world_size > 1:
            # Import the slicing logic (we'll add it as a helper method)
            start_idx, end_idx = self._get_slice_indices(total_items, world_size, rank)
            encoding_inputs = encoding_inputs[start_idx:end_idx]
            print(f"[MMEmbedClient] Rank {rank}/{world_size}: Processing items {start_idx} to {end_idx-1} (total: {end_idx-start_idx}/{total_items})")

        encoding_output, enc_metadata = self.encode_list_online(
            encoding_inputs,
            parameters
        )

        # Modify output paths if distributed encoding
        out_embeddings_path = file_info.out_embeddings_path
        out_index_path = file_info.out_index_path
        out_metadata_path = file_info.out_metadata_path

        if world_size > 1:
            out_embeddings_path = self._add_rank_suffix(out_embeddings_path, world_size, rank)
            out_index_path = self._add_rank_suffix(out_index_path, world_size, rank)
            if out_metadata_path:
                out_metadata_path = self._add_rank_suffix(out_metadata_path, world_size, rank)

        # Save embeddings
        torch.save(encoding_output.tensors, out_embeddings_path)
        print(f"[MMEmbedClient] Saved embeddings to {out_embeddings_path} (shape={encoding_output.tensors.shape})")

        # Save index
        IOUtils.write_json_file(encoding_output.index, out_index_path)
        print(f"[MMEmbedClient] Saved index to {out_index_path}")

        # Save metadata
        if out_metadata_path:
            IOUtils.write_json_file(enc_metadata.__dict__, out_metadata_path)
            print(f"[MMEmbedClient] Saved metadata to {out_metadata_path}")

        return


    def _get_slice_indices(
        self,
        total_items: int,
        world_size: int,
        rank: int
    ) -> Tuple[int, int]:
        """
        Calculate start and end indices for a uniform slice.
        Same logic as in local.py.
        """
        base_size = total_items // world_size
        remainder = total_items % world_size

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
        Same logic as in local.py.
        """
        directory = os.path.dirname(filepath)
        basename = os.path.basename(filepath)

        if '.' in basename:
            name, ext = basename.rsplit('.', 1)
            new_basename = f"{name}_world{world_size}_rank{rank}.{ext}"
        else:
            new_basename = f"{basename}_world{world_size}_rank{rank}"

        if directory:
            return os.path.join(directory, new_basename)
        else:
            return new_basename


if __name__ == "__main__":

    # Example usage - will automatically load host from config
    from src.model.mllm.utils.constants import LoadMode
    from src.model.embedder.embedders.mmembed._base import MMEmbed_Factory

    # Register the client implementation
    MMEmbed_Factory.register(LoadMode.REMOTE, MMEmbedClient)

    # Create factory and instantiate client
    factory = MMEmbed_Factory()
    mmembed = factory.create(mode=LoadMode.REMOTE)

    # Initialize remote model
    model_info = mmembed.initiate()
    print(f"Model info: {model_info}")

    # Test encoding queries (text-only)
    query_inputs = [
        EncodingInput(
            id = "q1",
            instruction = "retrieve an image relevant to the query",
            text = "a cat",
            image_paths = []
        ),
        EncodingInput(
            id = "q2",
            instruction = "retrieve an image relevant to the query",
            text = "a dog",
            image_paths = []
        )
    ]

    print("\n=== Encoding Queries ===")
    query_output, query_metadata = mmembed.encode_list_online(
        query_inputs,
        EncodingParameters(batch_size=2)
    )
    print(f"Query embeddings shape: {query_output.tensors.shape}")
    print(f"Query metadata: {query_metadata}")

    # Test encoding documents (images from test_files)
    doc_inputs = [
        EncodingInput(
            id="cat",
            text="",
            image_paths=["/root/omdr_workspace/src/model/test_files/cat.png"]
        ),
        EncodingInput(
            id="dog",
            text="",
            image_paths=["/root/omdr_workspace/src/model/test_files/dog.png"]
        ),
        # EncodingInput(
        #     id="erasers",
        #     text="",
        #     image_paths=["/root/omdr_workspace/src/model/test_files/erasers.png"]
        # ),
        # EncodingInput(
        #     id="house",
        #     text="",
        #     image_paths=["/root/omdr_workspace/src/model/test_files/house.png"]
        # ),
        # EncodingInput(
        #     id="pencil",
        #     text="",
        #     image_paths=["/root/omdr_workspace/src/model/test_files/pencil.png"]
        # )
    ]

    print("\n=== Encoding Documents ===")
    doc_output, doc_metadata = mmembed.encode_list_online(
        doc_inputs,
        EncodingParameters(batch_size=5)
    )
    print(f"Document embeddings shape: {doc_output.tensors.shape}")
    print(f"Document metadata: {doc_metadata}")

    # Compute similarity scores
    print("\n=== Computing Similarity Scores ===")
    import torch.nn.functional as F

    # Normalize embeddings for cosine similarity
    query_embeds_norm = F.normalize(query_output.tensors, p=2, dim=1)
    doc_embeds_norm = F.normalize(doc_output.tensors, p=2, dim=1)

    # Compute similarity matrix (queries x documents)
    similarity_matrix = torch.matmul(query_embeds_norm, doc_embeds_norm.T)

    # Print scores for each query
    query_ids = ["q1: a cat", "q2: a dog"]
    # doc_ids = ["cat", "dog", "erasers", "house", "pencil"]
    doc_ids = ["cat", "dog"]

    for i, query_id in enumerate(query_ids):
        print(f"\n{query_id}:")
        for j, doc_id in enumerate(doc_ids):
            score = similarity_matrix[i, j].item()
            print(f"  {doc_id}: {score:.4f}")

    # Clean up
    mmembed.free()
