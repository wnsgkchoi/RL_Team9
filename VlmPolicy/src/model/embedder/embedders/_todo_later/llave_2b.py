from __future__ import annotations

import os
import json
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from PIL import Image
from tqdm import tqdm

from embedders.base_embedder import BaseEmbedder

# ───────────────────────────────────────────────────────────────────────────────
# Model paths & constants – adjust if you move things
# ───────────────────────────────────────────────────────────────────────────────
LLAVE_PATH                  = "/mnt/sdc/jhyun/models/LLaVE-2B"
LLAVE_MODEL_NAME            = "llava_qwen"
LLAVE_CONV_TEMPLATE         = "qwen_1_5"
LLAVE_EMBED_DIM             = 3584
DEFAULT_RETRIEVAL_INSTR     = (
    "Retrieve passage, table or image (maybe with a caption) that provides an "
    "answer to the given query."
)

# Lazy-imported heavy deps
_llava_loaded = False

# ───────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ───────────────────────────────────────────────────────────────────────────────

def _lazy_import_llava():
    """Import heavy LLaVA modules only once and cache them globaly."""
    global _llava_loaded, conv_templates, tokenizer_image_token, process_images
    global IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, load_pretrained_model

    if not _llava_loaded:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # type: ignore
        from llava.conversation import conv_templates  # type: ignore
        from llava.mm_utils import tokenizer_image_token, process_images  # type: ignore
        from llava.model.builder import load_pretrained_model  # type: ignore
        _llava_loaded = True


# ───────────────────────────────────────────────────────────────────────────────
# Main wrapper
# ───────────────────────────────────────────────────────────────────────────────
class LLaVE_2B(BaseEmbedder):
    """MMEmbed-compatible facade around the LLaVE-2B dual-encoder."""

    # ---------------------------------------------------------------------
    # Construction & model lifecycle
    # ---------------------------------------------------------------------
    def __init__(self, cuda_num: int = 0, show_progress: bool = True):
        self.cuda_num = cuda_num
        self.device   = f"cuda:{cuda_num}"
        self.show_progress = show_progress

        self.tokenizer = None  # will be set in `load_model()`
        self.model = None
        self.image_processor = None

    def load_model(self) -> None:
        """Download / load weights, tokenizer & image pre-processing."""
        _lazy_import_llava()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (
                self.tokenizer,
                self.model,
                self.image_processor,
                _,
            ) = load_pretrained_model(
                LLAVE_PATH,
                None,
                model_name=LLAVE_MODEL_NAME,
                device_map=self.device,
            )
        self.model.eval()
        print("✅  LLaVE-2B loaded on", self.device)

    def delete_model(self) -> None:
        """Free GPU memory."""
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        torch.cuda.empty_cache()

    # ────────────────────────────────────────────────────────────────────
    # 1)  Text-only queries  (NO batching)
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode_queries(
        self,
        queries: List[str],
        output_filepath: str | None = None,
        instruction: Union[str, Sequence[str]] = DEFAULT_RETRIEVAL_INSTR,
        batch_size: int = 4,          # ← kept for API compatibility (ignored)
        max_length: int = 256,
    ) -> torch.Tensor:
        """
        Encode each query individually (no batching).

        * instruction: str or list[str] (same contract as before)
        * returns CPU tensor  (len(queries) × 3584)
        """
        if self.model is None:
            raise RuntimeError("Call `load_model()` first")
        _lazy_import_llava()

        # normalise instructions → list[str]
        if isinstance(instruction, str):
            instr_list = [instruction] * len(queries)
        else:
            if len(instruction) != len(queries):
                raise ValueError("`instruction` length must match `queries` length")
            instr_list = list(instruction)

        embeds: List[torch.Tensor] = []
        iterator = range(len(queries))
        if self.show_progress:
            iterator = tqdm(iterator, desc="EncodeQ-one-by-one")

        for i in iterator:
            instr = instr_list[i]
            q     = queries[i]

            conv = conv_templates[LLAVE_CONV_TEMPLATE].copy()  # type: ignore
            conv.append_message(conv.roles[0], f"{instr} {q}")
            conv.append_message(conv.roles[1], "\n")
            prompt = conv.get_prompt()

            ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            mask = ids.input_ids.ne(self.tokenizer.pad_token_id)

            try:
                emb = self.model.encode_multimodal_embeddings(
                    ids.input_ids, attention_mask=mask
                ).cpu()
            except Exception as e:
                print("❗ query failed –", e)
                emb = torch.zeros((1, LLAVE_EMBED_DIM), dtype=torch.float32)

            embeds.append(emb.squeeze(0))

        tensor = torch.stack(embeds, dim=0)
        if output_filepath:
            torch.save(tensor, output_filepath)
            print(f"💾 Query embeddings → {output_filepath}  shape={tensor.shape}")
        return tensor


    # ────────────────────────────────────────────────────────────────────
    # 2)  Multimodal corpus  (NO batching)
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode_corpus(
        self,
        data: List[Dict[str, Any]],
        out_embeddings_path: str,
        out_index_path: str,
        start_idx: int | None = None,
        end_idx: int | None = None,
        batch_size: int = 4,          # ← kept for API compatibility (ignored)
        max_length: int = 256,
    ) -> torch.Tensor:
        """
        Encode each corpus item one-by-one (no batching).  Output files are
        still redirected to  ./temp/<stem>_gpu<cuda>/  as before.
        """
        if self.model is None:
            raise RuntimeError("Call `load_model()` first")
        _lazy_import_llava()

        # ─── choose slice ───────────────────────────────────────────────
        total = len(data)
        start = 0 if start_idx is None else start_idx
        end   = total if end_idx is None else end_idx
        if start >= end or start >= total:
            raise ValueError("Invalid slice [start_idx, end_idx)")
        slice_data = data[start:end]

        # ─── encode per item ────────────────────────────────────────────
        embeds: List[torch.Tensor] = []
        idx_to_id: Dict[int, Any]  = {}

        iterator = enumerate(slice_data)
        if self.show_progress:
            iterator = tqdm(iterator, total=len(slice_data), desc="EncodeCorpus-one-by-one")

        for local_i, obj in iterator:
            idx_to_id[local_i] = obj["id"]

            text      = (obj["target"].get("text", "") or "").strip()
            img_paths = obj["target"].get("images", [])

            # try to open first valid image
            img = None
            for p in img_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    break
                except Exception:
                    continue

            conv = conv_templates[LLAVE_CONV_TEMPLATE].copy()  # type: ignore
            if img is not None:
                user_msg = (
                    DEFAULT_IMAGE_TOKEN
                    + " Represent the given image"
                    + (f", and this detail about the image: {text}" if text else ".")
                )
            else:
                user_msg = text
            conv.append_message(conv.roles[0], user_msg)
            conv.append_message(conv.roles[1], "\n")
            prompt = conv.get_prompt()

            # -------- encode ----------
            if img is None:                      # text-only
                ids  = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)
                mask = ids.input_ids.ne(self.tokenizer.pad_token_id)
                try:
                    emb = self.model.encode_multimodal_embeddings(
                        ids.input_ids, attention_mask=mask
                    ).cpu()
                except Exception:
                    emb = torch.zeros((1, LLAVE_EMBED_DIM), dtype=torch.float32)
            else:                                # image + text
                ids  = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(self.device)
                mask = ids.ne(self.tokenizer.pad_token_id)

                img_tensor = process_images([img], self.image_processor,
                                            self.model.config)
                img_tensor = [
                    t.to(dtype=torch.float16, device=self.device) for t in img_tensor
                ]
                try:
                    emb = self.model.encode_multimodal_embeddings(
                        ids,
                        attention_mask=mask,
                        images=img_tensor,
                        image_sizes=[img.size],
                    ).cpu()
                except Exception:
                    emb = torch.zeros((1, LLAVE_EMBED_DIM), dtype=torch.float32)

            embeds.append(emb.squeeze(0))

        tensor = torch.stack(embeds, dim = 0)

        # ─────────── persist artefacts ─────────────────────────────────
        embeddings_tensor = torch.stack(embeds, dim=0)

        # 1. slice-suffix (unchanged)
        stem, ext = os.path.splitext(out_embeddings_path)
        if start != 0 or end != total:
            out_embeddings_path = f"{stem}_{start}-{end}{ext}"
            out_index_path      = f"{os.path.splitext(out_index_path)[0]}_{start}-{end}.json"

        # 2. ENSURE DIRECTORY, BUT **DON'T** REDIRECT -------------------
        os.makedirs(os.path.dirname(out_embeddings_path), exist_ok=True)

        torch.save(embeddings_tensor, out_embeddings_path)
        with open(out_index_path, "w") as f:
            json.dump(idx_to_id, f, indent=4)

        print(f"💾 Saved embeddings → {out_embeddings_path}  shape={embeddings_tensor.shape}")
        return embeddings_tensor



# ───────────────────────────────────────────────────────────────────────────────
# CLI helper – mirrors *MMEmbed* usage
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    cli = argparse.ArgumentParser()
    cli.add_argument("--corpus_in_filepath", type=str, required=True)
    cli.add_argument("--out_embeddings_path", type=str, required=True)
    cli.add_argument("--out_index_path", type=str, required=True)
    cli.add_argument("--batch_size", type=int, default=4)
    cli.add_argument("--offset", type=int, default=0)
    cli.add_argument("--size", type=int, default=10000)
    cli.add_argument("--cuda", type=int, default=0)
    args = cli.parse_args()

    # 1) load corpus
    with open(args.corpus_in_filepath, "r", encoding="utf-8") as fh:
        corpus = json.load(fh)

    # 2) init + encode
    encoder = LLaVE_2B(cuda_num=args.cuda)
    encoder.load_model()

    encoder.encode_corpus(
        data=corpus,
        out_embeddings_path=args.out_embeddings_path,
        out_index_path=args.out_index_path,
        start_idx=args.offset,
        end_idx=args.offset + args.size,
        batch_size=args.batch_size,
    )
