

from __future__ import annotations

from typing import List, Tuple
import os
import time

import torch
from tqdm import tqdm

from model.embedder.embedders.nvembed_v2._base import NVEmbedV2
from model.embedder.data_structure.input import EncodingInput, EncodingParameters, OfflineEncodingInput
from model.embedder.data_structure.output import EncodingOutput, EncMetadata, TIME_MULTIPLIER_MS

import utils.file_io as IOUtils




class NVEmbedV2Local(NVEmbedV2):
    """Local inference wrapper for NV-Embed-v2 (text-only)."""


    def __init__(
        self,
        cuda_num: int | str = "auto",
        **kwargs
    ):
        super().__init__(**kwargs)

        self._cuda_num = cuda_num
        self._device_str = f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu'

        self._model = None

        # Default parameters
        self._default_parameters = EncodingParameters(
            show_progress   = True,
            batch_size      = 4,
            max_length      = 512,
            world_size      = 1,
            rank            = 0,
            write_file      = False,
            filepath        = None
        )

        return


    def initiate(self) -> None:
        """Load the NV-Embed-v2 model."""

        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(
            self._local_path,
            trust_remote_code = True
        )
        self._model.to(self._device_str)
        self._model.eval()  # Set to evaluation mode to ensure consistent behavior

        print()
        print(f"[NVEmbedV2Local] Model loaded successfully on {self._device_str}.")
        print()

        return


    def free(self) -> None:
        """Delete the model to free memory."""

        self._model = None

        print()
        print("[NVEmbedV2Local] Model deleted successfully.")
        print()

        return




    @torch.no_grad()
    def encode_online(
        self,
        input: EncodingInput,
        parameters: EncodingParameters = None
    ) -> Tuple[EncodingOutput, EncMetadata]:
        """
        Encode a single input.
        Delegates to encode_list_online for consistency.
        """

        # Warn if image inputs are provided
        self._warn_multimodal_inputs(input)

        encoding_output, enc_metadata = self.encode_list_online(
            [input],
            parameters
        )

        return encoding_output, enc_metadata




    @torch.no_grad()
    def encode_list_online(
        self,
        input_list: List[EncodingInput],
        parameters: EncodingParameters = None
    ) -> Tuple[EncodingOutput, EncMetadata]:
        """
        Encode a list of inputs (queries or documents).

        For queries: text with instruction
        For documents: text without instruction
        """

        if self._model is None:
            raise RuntimeError("Model not loaded. Call initiate() first.")

        # Apply default parameters
        if not parameters:
            parameters = EncodingParameters()
        parameters += self._default_parameters

        show_progress   = parameters.show_progress
        batch_size      = parameters.batch_size
        max_length      = parameters.max_length
        world_size      = parameters.world_size
        rank            = parameters.rank
        write_file      = parameters.write_file
        filepath        = parameters.filepath

        # Apply distributed slicing
        total_items = len(input_list)
        if world_size > 1:
            start_idx, end_idx = self._get_slice_indices(total_items, world_size, rank)
            input_list = input_list[start_idx:end_idx]
            print(f"[NVEmbedV2Local] Rank {rank}/{world_size}: Processing items {start_idx} to {end_idx-1} (total: {end_idx-start_idx}/{total_items})")

        # Initialize metadata
        metadata = EncMetadata(
            time                = 0,
            time_preprocess     = 0,
            time_infer          = 0,
            time_postprocess    = 0,
            num_inputs          = len(input_list),
            batch_size          = batch_size,
            batch_iterations    = 0
        )

        start_time = time.time()




        # Preprocessing: Prepare all inputs
        pre_time_start = time.time()

        all_texts = []
        id_list = []
        is_query_list = []
        instruction_list = []

        for enc_input in input_list:

            # Warn if images are provided (text-only model)
            self._warn_multimodal_inputs(enc_input)

            id_list.append(enc_input.id)

            # Determine if this is a query or document based on instruction presence
            has_instruction = enc_input.instruction is not None and enc_input.instruction != ""
            is_query = has_instruction
            is_query_list.append(is_query)

            # Get instruction
            instruction = enc_input.instruction if has_instruction else ""
            instruction_list.append(instruction)

            # Get text
            text = enc_input.text.strip() if enc_input.text else ""

            # Prepend instruction for queries
            if is_query:
                full_text = instruction + text
            else:
                full_text = text

            all_texts.append(full_text)

        metadata.time_preprocess = (time.time() - pre_time_start) * TIME_MULTIPLIER_MS




        # Inference: Batch encoding
        infer_time_start = time.time()

        all_embeddings = []
        num_batches = (len(all_texts) + batch_size - 1) // batch_size

        if show_progress:
            batch_iter = tqdm(
                range(0, len(all_texts), batch_size),
                desc = "Encoding",
                total = num_batches
            )
        else:
            batch_iter = range(0, len(all_texts), batch_size)

        for i in batch_iter:
            batch_texts = all_texts[i : i + batch_size]
            batch_is_query = is_query_list[i : i + batch_size]
            batch_instructions = instruction_list[i : i + batch_size]

            # Determine if this batch is all queries or all documents
            is_query_batch = batch_is_query[0]

            # Validate that all items in batch have the same type
            if not all(is_q == is_query_batch for is_q in batch_is_query):
                raise ValueError(
                    f"[NVEmbedV2Local] Mixed queries and documents in the same batch detected. "
                    f"Batch starting at index {i} contains both queries (with instructions) and documents (without instructions). "
                    f"Please encode queries and documents separately."
                )

            # Get instruction (use first one if all are consistent)
            instruction = batch_instructions[0] if all(inst == batch_instructions[0] for inst in batch_instructions) else ""

            try:
                # NV-Embed-v2 encode API
                embeddings = self._model.encode(
                    batch_texts,
                    instruction = instruction,
                    max_length = max_length
                )
            except Exception as e:
                print(f"[NVEmbedV2Local] Error during encoding: {e}")
                print(f"[NVEmbedV2Local] Returning zero embeddings for this batch.")
                # Fallback to zeros
                embeddings = torch.zeros((len(batch_texts), 4096), dtype=torch.float32).to(self._device_str)

            embeddings = embeddings.cpu()
            all_embeddings.append(embeddings)

            metadata.batch_iterations += 1

        metadata.time_infer = (time.time() - infer_time_start) * TIME_MULTIPLIER_MS




        # Postprocessing: Concatenate embeddings and build output
        post_time_start = time.time()

        # Concatenate all embeddings
        all_embeddings_tensor = torch.cat(all_embeddings, dim = 0)

        # Build index mapping
        index = {}
        for idx, id_val in enumerate(id_list):
            index[str(id_val)] = idx

        # Create output
        encoding_output = EncodingOutput(
            tensors = all_embeddings_tensor,
            index = index
        )

        metadata.time_postprocess = (time.time() - post_time_start) * TIME_MULTIPLIER_MS




        # Finalize metadata
        metadata.time = (time.time() - start_time) * TIME_MULTIPLIER_MS

        # Write to file if requested
        if write_file and filepath:
            torch.save(all_embeddings_tensor, filepath)

        return encoding_output, metadata


    @torch.no_grad()
    def encode_file_offline(
        self,
        file_info: OfflineEncodingInput,
        parameters: EncodingParameters = None
    ) -> None:
        """
        Encode passages from a file and save to disk.

        Reads from file_info.in_filepath (JSON/JSONL)
        Saves embeddings to file_info.out_embeddings_path (.pt)
        Saves index to file_info.out_index_path (.json)

        When world_size > 1, output files are suffixed with _world<world_size>_rank<rank>
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

        # Encode the inputs (slicing happens inside encode_list_online)
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
        print(f"[NVEmbedV2Local] Saved embeddings to {out_embeddings_path} (shape={encoding_output.tensors.shape})")

        # Save index
        IOUtils.write_json_file(encoding_output.index, out_index_path)
        print(f"[NVEmbedV2Local] Saved index to {out_index_path}")

        # Save metadata
        if out_metadata_path:
            IOUtils.write_json_file(enc_metadata.__dict__, out_metadata_path)
            print(f"[NVEmbedV2Local] Saved metadata to {out_metadata_path}")

        return




if __name__ == "__main__":

    # Test the NVEmbedV2Local implementation
    from src.model.mllm.utils.constants import LoadMode
    from src.model.embedder.embedders.nvembed_v2._base import NVEmbedV2_Factory
    import torch.nn.functional as F

    # Register the local implementation
    NVEmbedV2_Factory.register(LoadMode.LOCAL, NVEmbedV2Local)

    # Create factory and instantiate model
    factory = NVEmbedV2_Factory()
    nvembed = factory.create(mode=LoadMode.LOCAL, cuda_num=0)

    # Load model
    nvembed.initiate()

    # Test encoding queries (text-only)
    query_inputs = [
        EncodingInput(
            id = "q1",
            instruction = "Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
            text = "a cat sitting on a mat",
            image_paths = []
        ),
        EncodingInput(
            id = "q2",
            instruction = "Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
            text = "a dog playing in the park",
            image_paths = []
        )
    ]

    print("\n=== Encoding Queries ===")
    query_output, query_metadata = nvembed.encode_list_online(
        query_inputs,
        EncodingParameters(batch_size=2)
    )
    print(f"Query embeddings shape: {query_output.tensors.shape}")
    print(f"Query metadata: {query_metadata}")

    # Test encoding documents (text descriptions instead of images)
    doc_inputs = [
        EncodingInput(
            id="cat",
            text="A fluffy orange cat with green eyes",
            image_paths=[]
        ),
        EncodingInput(
            id="dog",
            text="A playful golden retriever running outdoors",
            image_paths=[]
        ),
        EncodingInput(
            id="erasers",
            text="A collection of colorful rubber erasers",
            image_paths=[]
        ),
        EncodingInput(
            id="house",
            text="A beautiful two-story suburban house with a garden",
            image_paths=[]
        ),
        EncodingInput(
            id="pencil",
            text="A yellow wooden pencil with an eraser on top",
            image_paths=[]
        )
    ]

    print("\n=== Encoding Documents ===")
    doc_output, doc_metadata = nvembed.encode_list_online(
        doc_inputs,
        EncodingParameters(batch_size=5)
    )
    print(f"Document embeddings shape: {doc_output.tensors.shape}")
    print(f"Document metadata: {doc_metadata}")

    # Compute similarity scores
    print("\n=== Computing Similarity Scores ===")

    # Normalize embeddings for cosine similarity
    query_embeds_norm = F.normalize(query_output.tensors, p=2, dim=1)
    doc_embeds_norm = F.normalize(doc_output.tensors, p=2, dim=1)

    # Compute similarity matrix (queries x documents)
    similarity_matrix = torch.matmul(query_embeds_norm, doc_embeds_norm.T)

    # Print scores for each query
    query_ids = ["q1: a cat sitting on a mat", "q2: a dog playing in the park"]
    doc_ids = ["cat", "dog", "erasers", "house", "pencil"]

    for i, query_id in enumerate(query_ids):
        print(f"\n{query_id}:")
        for j, doc_id in enumerate(doc_ids):
            score = similarity_matrix[i, j].item()
            print(f"  {doc_id}: {score:.4f}")

    # Clean up
    nvembed.free()
