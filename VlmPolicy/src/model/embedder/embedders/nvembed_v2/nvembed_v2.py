import os
import json
import argparse
from typing import List, Dict, Any, Union, Sequence

from tqdm import tqdm
import torch
import torch.nn.functional as F

from embedders.base_embedder import BaseEmbedder

# ───────────────────────────────────────────────────────────────────────────────
NVEMBED_PATH                  = "/mnt/sdc/jhyun/models/NV-Embed-v2"
NVEMBED_RETRIEVAL_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages that answer the question \n Query: "
)
NVEMBED_EMBED_DIM             = 4096        # model output size
# ───────────────────────────────────────────────────────────────────────────────


class NVEmbedV2(BaseEmbedder):
    """
    Text-only wrapper that follows the same public interface as MMEmbed/QQMM.
    """

    def __init__(
        self,
        cuda_num: int,
        show_progress: bool = True,
    ):
        self.device        = f"cuda:{cuda_num}"
        self.show_progress = show_progress
        self.model         = None

    # ────────────────────────────────────────────────────────────────────────
    # Model lifecycle
    # -----------------------------------------------------------------------

    def load_model(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(NVEMBED_PATH, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        print("NV-Embed-v2 loaded successfully.")

    def delete_model(self):
        self.model = None
        torch.cuda.empty_cache()
        print("Model deleted successfully.")

    # ────────────────────────────────────────────────────────────────────────
    # Query encoding  (API-compatible)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def encode_queries(
        self,
        queries: List[str],
        output_filepath: str | None = None,
        instruction: Union[str, Sequence[str]] = NVEMBED_RETRIEVAL_INSTRUCTION,
        batch_size: int = 4,
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Encode a list of query strings.

        Parameters match MMEmbed / QQMM:
        • `instruction` may be a single str or a list with the same length as `queries`.
          NV-Embed-v2 is text-only, so we simply *prefix* each query with its
          instruction before batching.  (If a list is provided we raise an error,
          as per-item batching would break the current highly-efficient code.)
        • Returns a *CPU* tensor of shape (N × 4096).
        """
        if isinstance(instruction, str):
            prefixed = [instruction + q for q in queries]
        else:
            # You can relax this if you ever need per-query instructions – it would
            # require falling back to one-by-one encoding.
            raise ValueError(
                "`NVEmbedV2.encode_queries` currently supports a single instruction "
                "string shared by all queries."
            )

        embeds: List[torch.Tensor] = []
        iterator = range(0, len(prefixed), batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Encoding queries")

        for i in iterator:
            batch = prefixed[i : i + batch_size]
            try:
                vecs = self.model.encode(
                    batch,
                    instruction=instruction,     # identical for the whole batch
                    max_length=max_length,
                )
            except Exception as e:
                print(f"[WARN] query batch failed – {e}")
                vecs = torch.zeros(
                    (len(batch), NVEMBED_EMBED_DIM), dtype=torch.float32, device=self.device
                )

            vecs = F.normalize(vecs, p=2, dim=1).cpu()
            embeds.append(vecs)

        tensor = torch.cat(embeds, dim=0)

        if output_filepath:
            torch.save(tensor, output_filepath)
            print(f"💾 Saved query embeddings → {output_filepath}  shape={tensor.shape}")

        return tensor

    # ────────────────────────────────────────────────────────────────────────
    # Corpus encoding (unchanged)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def encode_corpus(
        self,
        data: List[Dict[str, Any]],
        out_embeddings_path: str,
        out_index_path: str,
        start_idx: int | None,
        end_idx: int | None,
        batch_size: int = 4,
        max_length: int = 256,
    ):
        """
        Batched passage embedding for NV-Embed-v2 (text-only).
        The batching logic is identical to the original revision.
        """
        # ── slice corpus once ────────────────────────────────────────────────
        if start_idx is not None:
            end_idx = min(end_idx, len(data))
            data = data[start_idx:end_idx]

        texts: List[str] = []
        idx_to_data_id: Dict[int, Any] = {}
        for local_idx, item in enumerate(data):
            idx_to_data_id[local_idx] = item["id"]
            texts.append((item["target"]["text"] or "").strip())

        # ── batch-wise forward ───────────────────────────────────────────────
        all_embeddings: List[torch.Tensor] = []
        iterator = range(0, len(texts), batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Encoding corpus")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            try:
                vecs = self.model.encode(
                    batch_texts,
                    instruction="",          # passages need no instruction
                    max_length=max_length,
                )
            except Exception as e:
                print(f"[WARN] encode batch failed – {e}")
                vecs = torch.zeros(
                    (len(batch_texts), NVEMBED_EMBED_DIM),
                    dtype=torch.float32,
                    device=self.device,
                )

            vecs = F.normalize(vecs, p=2, dim=1).cpu()
            all_embeddings.append(vecs)

        all_embeddings = torch.cat(all_embeddings, dim=0)

        # ── file naming ──────────────────────────────────────────────────────
        if start_idx is not None:
            root, ext = os.path.splitext(out_embeddings_path)
            out_embeddings_path = f"{root}_{start_idx}-{end_idx}{ext}"
        os.makedirs(os.path.dirname(out_embeddings_path), exist_ok=True)

        torch.save(all_embeddings, out_embeddings_path)
        with open(out_index_path, "w") as f:
            json.dump(idx_to_data_id, f, indent=4)

        print(f"[INFO] Saved embeddings → {out_embeddings_path}  shape={all_embeddings.shape}")

        return all_embeddings

# ───────────────────────────────────────────────────────────────────────────────
# CLI runner – unchanged
# ───────────────────────────────────────────────────────────────────────────────
def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_in_filepath",  type=str, required=True)
    p.add_argument("--out_embeddings_path", type=str, required=True)
    p.add_argument("--out_index_path",      type=str, required=True)
    p.add_argument("--batch_size",          type=int, default=4)
    p.add_argument("--offset",              type=int, default=0)
    p.add_argument("--size",                type=int, default=10000)
    return p.parse_args()


def main():
    args = parse_arguments()

    # Load JSON list (same schema you already use)
    with open(args.corpus_in_filepath, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    encoder = NVEmbedV2(cuda_num=0)
    encoder.load_model()
    encoder.encode_corpus(
        data                = corpus,
        out_embeddings_path = args.out_embeddings_path,
        out_index_path      = args.out_index_path,
        start_idx           = args.offset,
        end_idx             = args.offset + args.size,
        batch_size          = args.batch_size,
        max_length          = 256,
    )

if __name__ == "__main__":
    main()
    
    
    


# import os
# import json
# import argparse
# from typing import List, Dict, Any

# from tqdm import tqdm
# import torch
# import torch.nn.functional as F

# from src.embedder.embedders.embedder import Embedder

# # ───────────────────────────────────────────────────────────────────────────────
# NVEMBED_PATH                  = "/mnt/sdc/jhyun/models/NV-Embed-v2"
# NVEMBED_RETRIEVAL_INSTRUCTION = (
#     "Instruct: Given a question, retrieve passages that answer the question \n Query: "
# )
# NVEMBED_EMBED_DIM             = 4096        # model output size
# # ───────────────────────────────────────────────────────────────────────────────


# class NVEmbedV2(Embedder):
#     """
#     Text‑only wrapper that follows the same interface as MMEmbed/LLaVEEmbed.
#     """

#     def __init__(
#         self,
#         cuda_num: int,
#         show_progress: bool = True,
#     ):
#         self.device        = f"cuda:{cuda_num}"
#         self.show_progress = show_progress

#         self.model = None
        
#         return

#     # ────────────────────────────────────────────────────────────────────────
#     # Model lifecycle
#     # -----------------------------------------------------------------------

#     def load_model(self):
#         from transformers import AutoModel
        
#         self.model = AutoModel.from_pretrained(NVEMBED_PATH, trust_remote_code=True)
#         self.model.to(self.device)
#         self.model.eval()
#         print("NV‑Embed‑v2 loaded successfully.")

#     def delete_model(self):
#         self.model = None
#         torch.cuda.empty_cache()
#         print("Model deleted successfully.")

#     # ────────────────────────────────────────────────────────────────────────
#     # Query encoding
#     # -----------------------------------------------------------------------

#     @torch.no_grad()
#     def encode_queries(
#         self, 
#         queries: List[str], 
#         output_filepath: str,
#         batch_size: int = 4,
#         max_length: int = 512
#     ):
#         """
#         Encode a list of query strings, prepend the retrieval instruction,
#         L2‑normalise, and save tensor (<n_queries> × 4096).
#         """
#         queries = [NVEMBED_RETRIEVAL_INSTRUCTION + q for q in queries]

#         embeds = []
#         iterator = range(0, len(queries), batch_size)
#         if self.show_progress:
#             iterator = tqdm(iterator, desc="Encoding queries")

#         for i in iterator:
#             batch = queries[i : i + batch_size]
#             try:
#                 vecs = self.model.encode(
#                     batch,
#                     instruction = NVEMBED_RETRIEVAL_INSTRUCTION,
#                     max_length = max_length,
#                 )
#             except Exception as e:
#                 print(f"[WARN] query batch failed – {e}")
#                 vecs = torch.zeros(
#                     (len(batch), NVEMBED_EMBED_DIM), dtype=torch.float32, device=self.device
#                 )

#             vecs = F.normalize(vecs, p=2, dim=1).cpu()
#             embeds.append(vecs)

#         embeds = torch.cat(embeds, dim=0)
#         if output_filepath is not None:
#             torch.save(embeds, output_filepath)
#             print(f"[INFO] Query embeddings → {output_filepath}  shape={embeds.shape}")
#         print(f"[INFO] Query embeddings → {output_filepath}  shape={embeds.shape}")
        
#         return embeds

#     # ────────────────────────────────────────────────────────────────────────
#     # Corpus encoding (JSON list identical to MMEmbed input)
#     # -----------------------------------------------------------------------

#     @torch.no_grad()
#     def encode_corpus(
#         self,
#         data: List[Dict[str, Any]],
#         out_embeddings_path: str,
#         out_index_path: str,
#         start_idx: int | None,
#         end_idx: int | None,
#         batch_size: int = 4,
#         max_length: int = 256,
#     ):
#         """
#         Batched passage embedding for NV‑Embed‑v2 (text‑only).
#         """
#         # ── slice corpus once ────────────────────────────────────────────────
#         if start_idx is not None:
#             end_idx = min(end_idx, len(data))
#             data = data[start_idx:end_idx]

#         texts: List[str] = []
#         idx_to_data_id: Dict[int, Any] = {}
#         for local_idx, item in enumerate(data):
#             idx_to_data_id[local_idx] = item["id"]
#             texts.append((item["target"]["text"] or "").strip())

#         # ── batch‑wise forward ───────────────────────────────────────────────
#         all_embeddings: List[torch.Tensor] = []
#         iterator = range(0, len(texts), batch_size)
#         if self.show_progress:
#             iterator = tqdm(iterator, desc="Encoding corpus")

#         for i in iterator:
#             batch_texts = texts[i : i + batch_size]
#             try:
#                 vecs = self.model.encode(
#                     batch_texts,
#                     instruction="",          # passages need no instruction
#                     max_length=max_length,
#                 )
#             except Exception as e:
#                 print(f"[WARN] encode batch failed – {e}")
#                 vecs = torch.zeros(
#                     (len(batch_texts), NVEMBED_EMBED_DIM),
#                     dtype=torch.float32,
#                     device=self.device,
#                 )

#             vecs = F.normalize(vecs, p=2, dim=1).cpu()
#             all_embeddings.append(vecs)

#         all_embeddings = torch.cat(all_embeddings, dim=0)

#         # ── file naming ──────────────────────────────────────────────────────
#         if start_idx is not None:
#             root, ext = os.path.splitext(out_embeddings_path)
#             out_embeddings_path = f"{root}_{start_idx}-{end_idx}{ext}"
#         os.makedirs(os.path.dirname(out_embeddings_path), exist_ok=True)

#         torch.save(all_embeddings, out_embeddings_path)
#         with open(out_index_path, "w") as f:
#             json.dump(idx_to_data_id, f, indent=4)

#         print(f"[INFO] Saved embeddings → {out_embeddings_path}  shape={all_embeddings.shape}")

#         return all_embeddings

# # ───────────────────────────────────────────────────────────────────────────────
# # CLI runner – consistent with the other encoders
# # ───────────────────────────────────────────────────────────────────────────────

# def parse_arguments():
#     p = argparse.ArgumentParser()
#     p.add_argument("--corpus_in_filepath",  type=str, required = True)
#     p.add_argument("--out_embeddings_path", type=str, required = True)
#     p.add_argument("--out_index_path",      type=str, required = True)
#     p.add_argument("--batch_size",          type=int, default = 4)
#     p.add_argument("--offset",              type=int, default = 0)
#     p.add_argument("--size",                type=int, default = 10000)
#     return p.parse_args()

# def main():
#     args = parse_arguments()

#     # Load JSON list (same schema you already use)
#     with open(args.corpus_in_filepath, "r", encoding="utf-8") as f:
#         corpus = json.load(f)

#     encoder = NVEmbedV2(cuda_num = 0)
#     encoder.load_model()
#     encoder.encode_corpus(
#         data                = corpus,
#         out_embeddings_path = args.out_embeddings_path,
#         out_index_path      = args.out_index_path,
#         start_idx           = args.offset,
#         end_idx             = args.offset + args.size,
#         batch_size          = args.batch_size,
#         max_length          = 256,
#     )

# if __name__ == "__main__":
#     main()

# # CUDA_VISIBLE_DEVICES=0 python3 src/embedder/nvembed_v2.py \
# #   --corpus_in_filepath  /mnt/sdc/jhyun/omdr_mountspace/datasets/_02_mmcoqa/embeddings/serializations/image+summary.json \
# #   --out_embeddings_path embed.pt \
# #   --out_index_path      embed.json \
# #   --offset 0 \
# #   --size   10000 \
# #   --batch_size 4







