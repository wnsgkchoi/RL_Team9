import os
import json
import argparse
from typing import List, Dict, Any, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Your existing embedder base class
from model.embedder.base_embedder import BaseEmbedder

# ---------------------------------------------------------------------------
# Re-implementation of ColPali (backbone + adapter) loading/inference logic.
# We adapt the code from "colpali.py" you shared, but embed everything here
# so we do NOT need `from m3docrag.retrieval import ColPaliRetrievalModel`.
# 
# This means we import from colpali_engine directly, as you originally showed:
#   from colpali_engine.models import ColPali, ColPaliProcessor
#   from colpali_engine.models import ColQwen2, ColQwen2Processor
#
# If you cannot import these in your environment, you must have them locally.
# ---------------------------------------------------------------------------
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    from colpali_engine.models import ColQwen2, ColQwen2Processor
except ImportError as e:
    raise ImportError(
        "Cannot import colpali_engine. Make sure you have it installed or available. "
        "Original error:\n" + str(e)
    )



def _init_colpali_model(
    backbone_name_or_path: str = "/job/model/colpaligemma-3b-pt-448-base",
    adapter_name_or_path: str  = "/job/model/colpali-v1.2",
    dtype: torch.dtype         = torch.bfloat16,
) -> (torch.nn.Module, Any):
    """
    Re-implement `init()` logic from your original colpali.py snippet.
    Loads the backbone + adapter, returning (model, processor).
    """

    # Decide if Qwen or standard ColPali
    # (some adapter_name might contain 'colqwen')
    kwargs = {}
    model_class      = ColPali
    processor_class  = ColPaliProcessor
    if "colqwen" in adapter_name_or_path:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        model_class      = ColQwen2
        processor_class  = ColQwen2Processor
        kwargs["attn_implementation"] = "flash_attention_2"

    # 1) load backbone
    model = model_class.from_pretrained(
        backbone_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **kwargs
    ).eval()

    # 2) load adapter
    model.load_adapter(adapter_name_or_path)

    # 3) processor
    processor = processor_class.from_pretrained(adapter_name_or_path)

    return model, processor



@torch.no_grad()
def _encode_images(
    model: torch.nn.Module,
    processor: Any,
    images: List[Image.Image],
    batch_size: int = 4,
    to_cpu: bool    = False,
    use_tqdm: bool  = False,
) -> List[torch.Tensor]:
    """
    Equivalent to `encode_images(...)` from your original colpali snippet,
    returning a list[Tensors], each shape = (n_tokens, emb_dim).
    """

    # If user does not supply a custom collate_fn, fallback to 'processor.process_images'
    collate_fn = getattr(processor, "process_images", None)
    if collate_fn is None:
        raise RuntimeError("Processor does not have a method `process_images`.")

    dataloader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    if use_tqdm:
        dataloader = tqdm(dataloader, desc="ColPali encode_images")

    doc_embs = []
    for batch_images in dataloader:
        # e.g. batch_images = {"pixel_values": ..., "attention_mask": ...}
        batch_images = {k: v.to(model.device) for k, v in batch_images.items()}
        embeddings = model(**batch_images)
        if to_cpu:
            embeddings = embeddings.to("cpu")
        # embeddings shape: (B, n_tokens, emb_dim)
        # We want a list of length B, each (n_tokens, emb_dim)
        emb_list = list(torch.unbind(embeddings, dim=0))
        doc_embs.extend(emb_list)

    return doc_embs



@torch.no_grad()
def _encode_queries(
    model: torch.nn.Module,
    processor: Any,
    queries: List[str],
    batch_size: int = 4,
    to_cpu: bool    = False,
    use_tqdm: bool  = False,
) -> List[torch.Tensor]:
    """
    Equivalent to `encode_queries(...)` from your original colpali snippet,
    returning a list[Tensors], each shape = (n_tokens, emb_dim).
    """

    collate_fn = getattr(processor, "process_queries", None)
    if collate_fn is None:
        raise RuntimeError("Processor does not have a method `process_queries`.")

    dataloader = DataLoader(
        queries,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    if use_tqdm:
        dataloader = tqdm(dataloader, desc="ColPali encode_queries")

    query_embs = []
    for batch_queries in dataloader:
        batch_queries = {k: v.to(model.device) for k, v in batch_queries.items()}
        embeddings = model(**batch_queries)
        if to_cpu:
            embeddings = embeddings.to("cpu")
        # embeddings shape: (B, n_tokens, emb_dim)
        emb_list = list(torch.unbind(embeddings, dim=0))
        query_embs.extend(emb_list)

    return query_embs




# ---------------------------------------------------------------------------
# Our main class, providing the same user-facing API as mmembed.py
# ---------------------------------------------------------------------------
class ColPali(BaseEmbedder):
    """
    A drop-in replacement for 'MMEmbed' that uses ColPali to embed images
    (ignoring text from corpus).  We produce one vector per doc by mean-pooling
    over the token dimension.

    - load_model() -> sets up the ColPali backbone+adapter
    - encode_queries(queries, ...) -> returns (Q, dim) CPU tensor
    - encode_corpus(data, ...) -> returns (N, dim) CPU tensor
    """

    def __init__(
        self,
        cuda_num: int = 0,
        show_progress: bool = True,
        backbone_model_path: str = "/job/model/colpaligemma-3b-pt-448-base",
        adapter_model_path: str  = "/job/model/colpali-v1.2",
    ):
        self.cuda_num            = cuda_num
        self.show_progress       = show_progress
        self.backbone_model_path = backbone_model_path
        self.adapter_model_path  = adapter_model_path

        self.model:  torch.nn.Module | None = None
        self.processor: Any          | None = None

    def initiate(self):
        """Load and prepare the ColPali backbone and adapter on the chosen GPU."""
        device_str = f"cuda:{self.cuda_num}" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = _init_colpali_model(
            backbone_name_or_path = self.backbone_model_path,
            adapter_name_or_path = self.adapter_model_path,
            dtype = torch.bfloat16
        )
        self.model.to(device_str)
        print(f"[ColPali] Model loaded to {device_str}")

    def free(self):
        """Release references so GPU memory can be freed."""
        self.model     = None
        self.processor = None
        print("[ColPali] Model deleted successfully.")

    @torch.no_grad()
    def encode_online(
        self,
        queries: List[str],
        output_filepath: str | None = None,
        instruction: Union[str, List[str]] = None,  # not used
        batch_size: int = 4,
        max_length: int = 256,  # not directly used
    ) -> torch.Tensor:
        """
        Encode queries into shape (Q, dim).  We call `_encode_queries` to get
        a list of shape (n_tokens, dim) for each query, then mean-pool.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load_model() before encode_queries().")

        # 1) get raw embeddings
        raw_embs = _encode_queries(
            model=self.model,
            processor=self.processor,
            queries=queries,
            batch_size=batch_size,
            to_cpu=False,
            use_tqdm=self.show_progress,
        )
        # 2) mean-pool per item
        # raw_embs is a list[T], each T shape = (n_tokens, dim)
        # We'll produce shape (Q, dim)
        pooled_list = []
        for emb_tensor in raw_embs:
            pooled = emb_tensor.mean(dim=0).cpu()
            pooled_list.append(pooled)
        final_tensor = torch.stack(pooled_list, dim=0)

        # 3) optional save
        if output_filepath:
            torch.save(final_tensor, output_filepath)
            print(f"[ColPali] Saved query embeddings → {output_filepath}  shape={final_tensor.shape}")

        return final_tensor

    @torch.no_grad()
    def encode_list_online(
        self,
        data: List[Dict[str, Any]],
        out_embeddings_path: str,
        out_index_path: str,
        start_idx: int | None,
        end_idx: int | None,
        batch_size: int = 4,
        max_length: int = 256,  # not used
    ) -> torch.Tensor:
        """
        Encode each data item (which has potential multiple images)
        by ignoring text and focusing on the first loadable image.
        Then we produce a single vector by mean-pooling across tokens.

        Saves:
          - The final embeddings to out_embeddings_path (possibly suffixed by start_idx-end_idx).
          - A JSON index mapping local_idx -> data_id to out_index_path.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load_model() before encode_corpus().")

        # slice if needed
        if start_idx is not None:
            end_idx = min(len(data), end_idx)
            subset = data[start_idx:end_idx]
        else:
            subset = data

        all_vectors = []
        idx_to_data_id = {}

        iterator = enumerate(subset)
        if self.show_progress:
            iterator = tqdm(iterator, desc="[ColPali] encode_corpus")
            
        # We'll build up a "batch" of images for colpali in smaller chunks
        # so we can reuse the colpali batch_size. We'll store them in an array,
        # run them, then clear. We'll also keep track of "positions" so we know
        # which doc each result belongs to.
        # 
        # Alternatively, to keep it simpler, we can just do a doc-by-doc approach,
        # but that won't leverage the batch advantage. We'll do doc-by-doc below
        # for clarity, though it's less efficient.
        for local_i, data_obj in iterator:
            # local_i is the index within subset
            data_id     = data_obj["id"]
            image_paths = data_obj["target"]["images"]

            # load the first accessible image
            chosen_image = None
            for img_path in image_paths:
                try:
                    chosen_image = Image.open(img_path).convert("RGB")
                    break
                except Exception as e:
                    print(f"[ColPali] Warning: failed to load image {img_path}: {e}")

            if chosen_image is None:
                # fallback zero vector
                # dimension unknown until model run, but let's guess e.g. 768
                zero_vec = torch.zeros(768, dtype=torch.float32)
                all_vectors.append(zero_vec)
                idx_to_data_id[local_i] = data_id
                continue

            # run colpali on exactly 1 doc's images => pass [chosen_image]
            raw_doc_embs = _encode_images(
                model=self.model,
                processor=self.processor,
                images=[chosen_image],
                batch_size=1,  # single doc
                to_cpu=False,
                use_tqdm=False,
            )

            if not raw_doc_embs:
                # fallback
                zero_vec = torch.zeros(768, dtype=torch.float32)
                all_vectors.append(zero_vec)
                idx_to_data_id[local_i] = data_id
                continue

            # raw_doc_embs[0] has shape (n_tokens, emb_dim)
            doc_vec = raw_doc_embs[0].mean(dim=0).cpu()  # shape (emb_dim,)
            all_vectors.append(doc_vec)
            idx_to_data_id[local_i] = data_id

        final_embeddings = torch.stack(all_vectors, dim=0)

        # naming for out_emb path
        if start_idx is not None:
            base, ext = os.path.splitext(out_embeddings_path)
            out_emb_path = f"{base}_{start_idx}-{end_idx}{ext}"
        else:
            out_emb_path = out_embeddings_path

        torch.save(final_embeddings, out_emb_path)
        with open(out_index_path, "w", encoding="utf-8") as f:
            json.dump(idx_to_data_id, f, indent=4)

        print(
            f"[ColPali] encode_corpus: saved embeddings to {out_emb_path} "
            f"(shape={final_embeddings.shape}), index -> {out_index_path}"
        )

        return final_embeddings
