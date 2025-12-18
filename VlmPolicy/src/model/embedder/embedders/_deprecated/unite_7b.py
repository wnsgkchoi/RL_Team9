#!/usr/bin/env python3
# unite.py
# Author: Joohyung Yun – DSLab, POSTECH
#
# Wrapper around *Unite-Instruct-Qwen2-VL-7B* that matches the Embedder API
# implemented by mmembed.py.  Text-only queries are handled one-by-one exactly
# like the new version of MM-Embed; corpus encoding remains batched.

from __future__ import annotations

import os
import json
from typing import List, Dict, Sequence, Union

from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor

from qwen_vl_utils import process_vision_info
from modeling_unite import UniteQwen2VL
from embedders.base_embedder import BaseEmbedder   # base class

# ---------------------------------------------------------------------------

UNITE_MODEL_ID = "/mnt/sdc/jhyun/models/Unite-7B"
MULTI_MODAL_INSTRUCTION = (
    "Retrieve passage, table, image or video that answers the given query."
)

class UniteEmbed(BaseEmbedder):
    """
    A plug-compatible replacement for MMEmbed that produces 3 584-d embeddings
    with Unite-Qwen2-VL-7B.

    Example
    -------
    >>> model = UniteEmbed(cuda_num=0)
    >>> model.load_model()
    >>> vec = model.encode_queries(['Hello world'])[0]
    >>> print(vec.shape)   # torch.Size([3584])
    """

    def __init__(
        self,
        cuda_num: int = 0,
        show_progress: bool = True,
    ) -> None:
        self.cuda_num = cuda_num
        self.show_progress = show_progress

        self.model = None           # filled by load_model()
        self.tokenizer = None
        self.processor = None

        # number of features per sample
        self.embed_dim = 3584

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def load_model(self):
        """Load Unite-Qwen2-VL-7B and its tokenizer / processor."""
        device = f"cuda:{self.cuda_num}"
        self.model = UniteQwen2VL.from_pretrained(
            UNITE_MODEL_ID,
            device_map=device,
            torch_dtype=torch.bfloat16,
            # Uncomment for FA2 if your GPU / environment supports it
            # attn_implementation="flash_attention_2",
        )
        # 1) (Unite) keep tokenizer / processor lines
        self.tokenizer  = AutoTokenizer.from_pretrained(UNITE_MODEL_ID, use_fast=False)
        self.processor  = AutoProcessor.from_pretrained(
            UNITE_MODEL_ID,
            min_pixels=256*28*28,
            max_pixels=1280*28*28,
        )
        print("UNITE model loaded successfully.")
        
        # 2) ***HOT-FIX for LLaVA-Next style models (MM-Embed, Unite-7B, etc.)***
        #    Newer Transformers require these two attributes to live on the processor.
        # v_cfg = getattr(self.model.config, "vision_config", None)
        # if v_cfg and not hasattr(self.processor, "patch_size"):
        #     self.processor.patch_size = v_cfg.patch_size                 # usually 14:contentReference[oaicite:1]{index=1}
        # if not hasattr(self.processor, "vision_feature_select_strategy"):
        #     # many checkpoints set it in the root config; fall back to "full"
        #     strategy = getattr(self.model.config, "vision_feature_select_strategy", "full")
        #     self.processor.vision_feature_select_strategy = strategy

        # # 3) *optional but safer* – make sure the model config itself also owns the field
        # if not hasattr(self.model.config, "vision_feature_select_strategy"):
        #     self.model.config.vision_feature_select_strategy = "full"

        # print("Model & processor patched for vision_feature_select_strategy.")

    def delete_model(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        print("UNITE model deleted successfully.")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _build_inputs(
        self,
        messages: List[dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a Unite chat *messages* list (see official examples) into
        processed tensor dict ready for model(**inputs).
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "<|endoftext|>"

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(f"cuda:{self.cuda_num}")

    # ------------------------------------------------------------------
    # Text-only queries (one-by-one, like new MMEmbed)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_queries(
        self,
        queries: List[str],
        output_filepath: str | None = None,
        instruction: Union[str, Sequence[str]] = MULTI_MODAL_INSTRUCTION,
        batch_size: int = 4,      # ignored, kept for API compatibility
        max_length: int = 256,    # ignored (handled inside processor)
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Call load_model() first")

        # normalise instruction list
        if isinstance(instruction, str):
            instr_list = [instruction] * len(queries)
        else:
            if len(instruction) != len(queries):
                raise ValueError("Instruction length must match queries length")
            instr_list = list(instruction)

        iterator = range(len(queries))
        if self.show_progress:
            iterator = tqdm(iterator, desc="Unite-encodeQ")

        embs: List[torch.Tensor] = []
        for i in iterator:
            q, instr = queries[i], instr_list[i]

            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": q},
                    {"type": "text", "text": instr},
                ]},
            ]
            try:
                inputs = self._build_inputs(messages)
                emb = self.model(**inputs).squeeze(0).cpu()  # [3584]
            except Exception as e:
                print(f"❗ Unite encoding failed for query {i}: {e}")
                emb = torch.zeros(self.embed_dim)

            embs.append(emb)

        tensor = torch.stack(embs)

        if output_filepath:
            torch.save(tensor, output_filepath)
            print(f"💾 Saved embeddings → {output_filepath} shape={tensor.shape}")

        return tensor

    # ------------------------------------------------------------------
    # Multimodal corpus encoding (text + optional image / video)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_corpus(
        self,
        data: List[dict],
        out_embeddings_path: str,
        out_index_path: str,
        start_idx: int | None,
        end_idx: int | None,
        batch_size: int = 1,          # 70 B GPU-heavy; default to 1
        max_length: int = 256,        # ignored
    ) -> torch.Tensor:
        """
        Each `data` item:

            {
              "id": "...",
              "target": {
                 "text": "...",
                 "images": ["path_or_url", ...],   # may be empty
                 "videos": ["path_or_url", ...],   # optional, same keys as Unite
              }
            }

        Saves:
          • embeddings tensor → out_embeddings_path
          • idx-to-id mapping  → out_index_path
        """
        if self.model is None:
            raise RuntimeError("Call load_model() first")

        # slice range if requested
        if start_idx is not None:
            data = data[start_idx:end_idx]

        idx_to_id: Dict[int, str] = {}
        all_embs: List[torch.Tensor] = []
        iterable = range(0, len(data), batch_size)
        if self.show_progress:
            iterable = tqdm(iterable, desc="Unite-encodeCorpus")

        for offset in iterable:
            batch = data[offset: offset + batch_size]
            batch_inputs = []
            for item in batch:
                idx_to_id[offset] = item["id"]

                txt  = item["target"]["text"].strip()
                imgs = item["target"].get("images", [])
                # vids = item["target"].get("videos", [])

                content: List[dict] = [{"type": "text", "text": txt}]
                if imgs:
                    content.insert(0, {"type": "image", "image": imgs[0]})

                content.append({"type": "text", "text": MULTI_MODAL_INSTRUCTION})
                batch_inputs.append(
                    self._build_inputs([{"role": "user", "content": content}])
                )

            # unite can't batch mixed modalities easily, so run per sample
            embed_batch = []
            for inp in batch_inputs:
                try:
                    emb = self.model(**inp).squeeze(0).cpu()
                except Exception as e:
                    print(f"❗ Unite corpus encoding error: {e}")
                    emb = torch.zeros(self.embed_dim)
                embed_batch.append(emb)
            all_embs.extend(embed_batch)

        tensor = torch.stack(all_embs)
        torch.save(tensor, out_embeddings_path)
        with open(out_index_path, "w") as f:
            json.dump(idx_to_id, f, indent=2)

        print(f"💾 Saved embeddings → {out_embeddings_path} shape={tensor.shape}")
        return tensor


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. build wrapper and load UNITE
    unite = UniteEmbed(cuda_num=0, show_progress=True)
    unite.load_model()

    # 2. encode two short text queries
    queries = [
        "A dog playing with a ball in the park.",
        "A happy puppy chasing a toy outdoors.",
    ]
    vecs = unite.encode_queries(queries)
    print("Embeddings tensor shape :", vecs.shape)          # (2, 3584)

    # 3. quick cosine-similarity sanity check
    sim = torch.nn.functional.cosine_similarity(vecs[0], vecs[1], dim=0)
    print("Cosine similarity between the two queries:", float(sim))

    # 4. clean up GPU memory (optional)
    unite.delete_model()