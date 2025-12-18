import os
import json
import torch
from PIL import Image
from typing import List, Union, Sequence, Dict, Optional

from tqdm import tqdm

# Your custom modules for Qwen2-VL
import sys
sys.path.insert(1, '/root/omdr_workspace/algorithms/VLM2Vec')
from model import MMEBModel
from arguments import ModelArguments
from model_utils import load_processor, QWEN2_VL, vlm_image_tokens

# Same parent class as MMEmbed
from embedders.base_embedder import BaseEmbedder

# This instruction is mostly a placeholder to match the signature of MMEmbed
MODALITY_AGNOSTIC_MMEMBED_RETRIEVAL_INSTRUCTION = (
    "Retrieve passage, table or image (maybe with a caption) that provides an answer to the given query."
)

class VLM2Vec_7B(BaseEmbedder):
    """
    A naive, drop-in replacement for MMEmbed, but built on top of Qwen2-VL 
    + LoRA-based “MMEBModel”.  No batching or max_length control is done 
    here beyond single-item loops.
    """

    def __init__(
        self, 
        cuda_num: int = 0, 
        show_progress: bool = True,
        # For clarity, pass needed model_args here or let the user set them externally
        model_args: ModelArguments = None,
    ):
        self.mode = None
        self.cuda_num = cuda_num
        self.show_progress = show_progress

        if model_args is None:
            # An example of how you might set a default
            # Adjust as you see fit
            model_args = ModelArguments(
                model_name="/mnt/sdc/jhyun/models/Qwen2-VL-7B-Instruct",
                checkpoint_path="/mnt/sdc/jhyun/models/VLM2Vec-Qwen2VL-7B",
                pooling="last",
                normalize=True,
                model_backbone="qwen2_vl",
                lora=True
            )
        self.model_args = model_args

        # Pre-load processor. Typically we do it once.
        self.processor = load_processor(self.model_args)

        self.model = None
        

    def load_model(self) -> None:
        """
        Loads the Qwen2-VL-based model (via MMEBModel) onto GPU.
        """
        self.model = MMEBModel.load(self.model_args)
        self.model = self.model.to(f"cuda:{self.cuda_num}", dtype=torch.bfloat16)
        self.model.eval()
        print("Model loaded successfully.")


    def delete_model(self) -> None:
        """
        De-initialize the model to free up GPU.
        """
        self.model = None
        print("Model deleted successfully.")


    @torch.no_grad()
    def encode_queries(
        self,
        queries: List[str],
        output_filepath: str | None = None,
        instruction: Union[str, Sequence[str]] = MODALITY_AGNOSTIC_MMEMBED_RETRIEVAL_INSTRUCTION,
        batch_size: int = 4,      # ignored in this naive implementation
        max_length: int = 256,    # ignored in this naive implementation
    ) -> torch.Tensor:
        """
        Encode each query (text-only) one by one. 
        Naive approach: no real batching, no max_length handling.

        Returns a CPU tensor of shape (len(queries), EMB_DIM).
        If `output_filepath` is provided, also saves with torch.save().
        """
        if self.model is None:
            raise RuntimeError("Call `load_model()` before `encode_queries()`")

        # If user passed a single string for instruction, broadcast it
        if isinstance(instruction, str):
            instr_list = [instruction] * len(queries)
        else:
            if len(instruction) != len(queries):
                raise ValueError("`instruction` length must match `queries` length")
            instr_list = list(instruction)

        embeddings = []
        iterator = range(len(queries))
        if self.show_progress:
            iterator = tqdm(iterator, desc="VLM2Vec: encode_queries")

        for i in iterator:
            text_query = queries[i]
            instr = instr_list[i]

            # For text queries, we just pass them to the processor
            # This is a naive example. 
            # You might choose to incorporate the instruction into the text if you want.
            inputs = self.processor(
                text=text_query,
                images=None,
                return_tensors="pt"
            )
            inputs = {k: v.to(f"cuda:{self.cuda_num}") for k, v in inputs.items()}

            # QWEN2-VL usage: "qry=" yields `qry_reps`
            try:
                out = self.model(qry=inputs)["qry_reps"]  # shape (1, embed_dim)
                emb = out.squeeze(0).cpu()
            except Exception as e:
                print(f"❗ Encoding failed for query '{text_query}': {e} – using zeros")
                # This is a fallback if something goes wrong
                emb_dim = getattr(self.model, "output_dim", 3584)
                emb = torch.zeros(emb_dim, dtype=torch.float32)

            embeddings.append(emb)

        tensor = torch.stack(embeddings, dim=0)

        # Optional save
        if output_filepath:
            torch.save(tensor, output_filepath)
            print(f"💾 Saved query embeddings → {output_filepath}  shape={tensor.shape}")

        return tensor




    def encode_corpus(
        self,
        data: List[Dict[str, Union[str, Dict]]],
        out_embeddings_path: str,
        out_index_path: str,
        start_idx: Optional[int],
        end_idx: Optional[int],
        batch_size: int = 4,      # ignored in this naive implementation
        max_length: int = 256,    # ignored in this naive implementation
    ) -> torch.Tensor:
        """
        Encode a corpus of items. Each item has:
          data_obj['id']       -> the ID of the item
          data_obj['target']   -> { "text": ..., "images": [...] }

        We build a single embedding from `text + first-loadable-image` 
        (if available). No real batching or max_length done.

        Saves embeddings to `out_embeddings_path`, and an index file to `out_index_path`.
        Returns the embeddings (CPU).
        """
        if self.model is None:
            raise RuntimeError("Call `load_model()` before `encode_corpus()`")

        n_total = len(data)

        # ── decide which part of `data` to embed ─────────────────────
        slice_start = 0 if start_idx is None else start_idx
        slice_end   = n_total if end_idx is None else min(n_total, end_idx)
        subset      = data[slice_start:slice_end]

        idx_to_data_id = {}

        embeddings_list = []
        iterator = range(len(subset))
        if self.show_progress:
            iterator = tqdm(iterator, desc="VLM2Vec: encode_corpus")

        for i in iterator:
            data_obj = subset[i]
            data_id = data_obj["id"]
            if start_idx is None:         # whole slice → local indices 0,1,2…
                idx_to_data_id[i] = data_id
            else:                         # global offset
                idx_to_data_id[i + slice_start] = data_id

            # Handle text
            text_str = data_obj["target"]["text"].strip()

            # Handle images (load first successful)
            img = None
            image_paths = data_obj["target"].get("images", [])
            for path in image_paths:
                try:
                    img_candidate = Image.open(path).convert("RGB")
                    img = img_candidate
                    break  # first successful
                except Exception as e:
                    print(f"Warning: failed to load image at {path}: {e}")
            if img is not None:
                text_str = f"{vlm_image_tokens[QWEN2_VL]} {text_str}"

            # Prepare inputs for the Qwen2-VL model
            # For corpus, we feed them as "tgt" to produce `tgt_reps`.
            try:
                inputs = self.processor(
                    text=text_str,
                    images=img,
                    return_tensors="pt"
                )
                inputs = {k: v.to(f"cuda:{self.cuda_num}") for k, v in inputs.items()}
                
                for k in ("pixel_values", "image_grid_thw", "image_patches"):
                    if k in inputs and inputs[k].ndim < 4:      # e.g. 3-D -> 4-D
                        inputs[k] = inputs[k].unsqueeze(0)
                
                out = self.model(tgt=inputs)["tgt_reps"]  # shape (1, embed_dim)
                emb = out.squeeze(0).cpu()
            except Exception as e:
                print(f"❗ Encoding failed for corpus item {data_id}: {e} – using zeros")
                emb_dim = getattr(self.model, "output_dim", 3584)
                emb = torch.zeros(emb_dim, dtype=torch.float32)

            embeddings_list.append(emb)

        # Combine all embeddings
        all_embeddings = torch.stack(embeddings_list, dim=0)

        # Save embeddings
        # Insert `_start-end` in the filename if desired
        if start_idx is None:                       # no suffix
            final_embeddings_path = out_embeddings_path
        else:                                       # add _start-end
            base, ext = os.path.splitext(out_embeddings_path)
            final_embeddings_path = f"{base}_{slice_start}-{slice_end}{ext}"

        torch.save(all_embeddings, final_embeddings_path)
        print(f"💾 Saved corpus embeddings → {final_embeddings_path}  shape={all_embeddings.shape}")

        # Save the index mapping
        with open(out_index_path, "w", encoding="utf-8") as f:
            json.dump(idx_to_data_id, f, indent=4)
        print(f"💾 Saved corpus index → {out_index_path}")

        return all_embeddings



if __name__ == "__main__":
    
    import sys
    # sys.path.insert(1, '/root/omdr_workspace/Algorithms/Baseline/ColBERT')
    sys.path.insert(1, '/root/omdr_workspace/algorithms/VLM2Vec')


    from src.model import MMEBModel
    from src.arguments import ModelArguments
    from src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens
    from PIL import Image
    import torch

    model_args = ModelArguments(
        model_name='/mnt/sdc/jhyun/models/Qwen2-VL-7B-Instruct',
        checkpoint_path='/mnt/sdc/jhyun/models/VLM2Vec-Qwen2VL-7B',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )

    processor = load_processor(model_args)
    model = MMEBModel.load(model_args)
    model = model.to('cuda', dtype=torch.bfloat16)
    model.eval()

    # Image + Text -> Text
    inputs = processor(text=f'{vlm_image_tokens[QWEN2_VL]} Represent the given image with the following question: What is in the image',
                       images=Image.open('/root/omdr_workspace/cat_dog.webp'),
                       return_tensors="pt")


    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)
    qry_output = model(qry=inputs)["qry_reps"]

    string = 'A cat and a dog'
    inputs = processor(text=string,
                       images=None,
                       return_tensors="pt")
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    tgt_output = model(tgt=inputs)["tgt_reps"]
    print(string, '=', model.compute_similarity(qry_output, tgt_output))
    ## A cat and a dog = tensor([[0.3301]], device='cuda:0', dtype=torch.bfloat16)

    string = 'A cat and a tiger'
    inputs = processor(text=string,
                       images=None,
                       return_tensors="pt")
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    tgt_output = model(tgt=inputs)["tgt_reps"]
    print(string, '=', model.compute_similarity(qry_output, tgt_output))
    ## A cat and a tiger = tensor([[0.2891]], device='cuda:0', dtype=torch.bfloat16)