

# src/embedder/jasper.py
import os
import json
import argparse
from typing import List, Dict, Any, Union, Sequence

from tqdm import tqdm
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from embedders.base_embedder import BaseEmbedder


# ───────────────────────────────────────────────────────────────────────────────
JASPER_PATH                  = "/mnt/sdc/jhyun/models/jasper_en_vision_language_v1"
JASPER_RETRIEVAL_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages that answer the question \n Query: "
)
JASPER_EMBED_DIM             = 1024          # vector_dim we request from Jasper
# ───────────────────────────────────────────────────────────────────────────────


class Jasper(BaseEmbedder):
    """
    Text-only wrapper around *jasper_en_vision_language_v1* that mirrors the API
    of NVEmbedV2.  (Image support is intentionally disabled.)
    """

    # ────────────────────────────────────────────────────────────────────────
    # Construction
    # -----------------------------------------------------------------------
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
        """
        Loads the Jasper SentenceTransformer and sets the maximum sequence
        length to 1 024 tokens for faster inference.
        """
        self.model = SentenceTransformer(
            JASPER_PATH,
            trust_remote_code=True,
            device=self.device,
            model_kwargs={
                "torch_dtype": torch.float32,
                "attn_implementation": "sdpa",
            },
            # The text encoder already outputs 1024-d vectors; we turn off image
            # pathways via `is_text_encoder=False`.
            config_kwargs={"is_text_encoder": False, "vector_dim": JASPER_EMBED_DIM},
        )
        self.model.max_seq_length = 1024
        self.model.eval()
        print("Jasper (v1) loaded successfully.")

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
        instruction: Union[str, Sequence[str]] = JASPER_RETRIEVAL_INSTRUCTION,
        batch_size: int = 4,
        max_length: int = 1024,
    ) -> torch.Tensor:
        """
        Encodes queries and returns an *L2-normalised* CPU tensor
        (N × 1024).  Jasper expects `prompt_name="s2p_query"` for
        retrieval-style embeddings; we apply that under the hood.
        """
        if isinstance(instruction, str):
            # Keep behaviour identical to NVEmbedV2 (prefix once for the batch)
            prefixed = [instruction + q for q in queries]
        else:
            raise ValueError(
                "`Jasper.encode_queries` currently supports a single instruction "
                "string shared by all queries."
            )

        vecs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            prompt_name="s2p_query",
            normalize_embeddings=True,        # built-in L2 normalisation
            convert_to_tensor=True,
            max_length=max_length,
            show_progress_bar=self.show_progress,
        ).cpu()                              # ensure result on CPU

        if output_filepath:
            torch.save(vecs, output_filepath)
            print(f"💾 Saved query embeddings → {output_filepath}  shape={vecs.shape}")

        return vecs

    # ────────────────────────────────────────────────────────────────────────
    # Corpus encoding (identical schema to NVEmbedV2)
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
        max_length: int = 1024,
    ):
        """
        Encodes passage texts only (images are ignored by design).
        Saves:
          • `<out_embeddings_path>` – torch.float32 (N × 1024) tensor
          • `<out_index_path>`      – JSON map {local_idx: original_id}
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

        # ── bulk forward pass (SentenceTransformer batches internally) ──────
        all_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            prompt_name=None,
            normalize_embeddings=True,
            convert_to_tensor=True,
            max_length=max_length,
            show_progress_bar=self.show_progress,
        ).cpu()

        # ── file naming & saving ────────────────────────────────────────────
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
# CLI runner – identical flags to NVEmbedV2 for drop-in use
# ───────────────────────────────────────────────────────────────────────────────
def _parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_in_filepath",  type=str, required=True)
    p.add_argument("--out_embeddings_path", type=str, required=True)
    p.add_argument("--out_index_path",      type=str, required=True)
    p.add_argument("--batch_size",          type=int, default=4)
    p.add_argument("--offset",              type=int, default=0)
    p.add_argument("--size",                type=int, default=10000)
    return p.parse_args()


def _main():
    args = _parse_arguments()

    with open(args.corpus_in_filepath, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    encoder = Jasper(cuda_num=0)
    encoder.load_model()
    encoder.encode_corpus(
        data                = corpus,
        out_embeddings_path = args.out_embeddings_path,
        out_index_path      = args.out_index_path,
        start_idx           = args.offset,
        end_idx             = args.offset + args.size,
        batch_size          = args.batch_size,
    )






if __name__ == "__main__":


    DOC1 = """
    Blue light is scattered in all directions by the tiny molecules of air in Earth's atmosphere. 
    Blue is scattered more than other colors because it travels as shorter, smaller waves. This is why we see a blue sky most of the time. 
    Closer to the horizon, the sky fades to a lighter blue or white.
    """
    DOC2 = """
    When choosing colors, you can consider the following factors:
    Color theory: Understand how colors work together and how they can evoke different reactions. 
    Color psychology: Consider how colors affect emotions, behaviors, and responses. 
    Brand identity: Colors can convey meaning and information about a brand. 
    Mood: Consider the mood you want to create. For example, brighter colors can feel cheerful, while cooler colors can be calming.
    Space: Consider the size of the space and the amount of natural light it receives. Dark colors can make a room feel smaller, while light colors can make it feel larger.
    Color wheel: Use the color wheel to identify primary, secondary, and tertiary colors. 
    Color combinations: Decide how to best complement your preferred color with others. 
    Color palette: Limit your color palette to a main color and one or two additional colors. 
    60-30-10 rule: Use a primary color 60% of the time, a secondary color 30% of the time, and an accent color 10% of the time
    """
    if __name__ == "__main__":
        # load model
        use_gpu = False
        model_name = "/mnt/sdc/jhyun/models/jasper_en_vision_language_v1"
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device="cpu" if not use_gpu else "cuda",
            model_kwargs={
                "torch_dtype": torch.bfloat16 if use_gpu else torch.float32,
                "attn_implementation": "sdpa"
            },
            # vector_dim must be 12288, 1024, 512, 256
            ## 1024 is recommended
            # set is_text_encoder 'True', if you do not encode image
            config_kwargs={"is_text_encoder": False, "vector_dim": 1024},
        )
        # We can reduce the max_seq_length from the default of 2048 for faster encoding
        model.max_seq_length = 1024

        # data
        q_list = [
            "Why the sky is blue?",
            "how to choose suitable color",
        ]
        doc_list = [
            DOC1,
            [{"type": "image_path", "content": "./assets/img1.png"}, {"type": "text", "content": "Hope this image helps!"}],
            DOC2,
            [{"type": "image_path", "content": "./assets/img2.png"}],
        ]
        q_vecs = model.encode(q_list, prompt_name="s2p_query")
        doc_vecs = model.encode(doc_list)

        # calculate similarity
        similarities = model.similarity(q_vecs, doc_vecs)
        print(similarities)
        # the output is:
        # tensor([[0.7775, 0.7594, 0.2429, 0.2187],
        #         [0.3226, 0.3054, 0.7421, 0.5484]])