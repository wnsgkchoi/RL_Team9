# qqmm_v1.py

import os
import json
import torch
from PIL import Image
from typing import List, Union, Sequence, Optional

# If your project structure needs a relative import from your local embedder base:
# from src.embedder.embedders.embedder import Embedder
# Else just define an empty base or remove inheritance as needed.
from embedders.base_embedder import BaseEmbedder

# ------------------------------------------------------------------
# If you need to ensure QQMM is on the path:
import sys
sys.path.insert(1, '/root/omdr_workspace/algorithms/QQMM-embed')

# QQMM imports
from qqmm.models import build_processor
from qqmm.utils.parameter_manage import Parameters
from qqmm.models.qqmm_nav_qwen2.modeling_qqmm import QQMMForCausalLM
from qqmm.utils.chat import EmbedBot

# If you want a default instruction in parallel to MMEmbed:
MODALITY_AGNOSTIC_QQMM_INSTRUCTION = (
    "Represent the given text or image for retrieval."
)

# Path to your QQMM config & model
QQMM_CONFIG_YAML = "/root/omdr_workspace/algorithms/QQMM-embed/configs/embed/qqmm-embed/mmeb.yaml"
QQMM_MODEL_PATH  = "/mnt/sdc/jhyun/models/QQMM-embed-v1"

# By observation from your logs: embed size is 3584
QQMM_EMBED_DIM = 3584


class QQMM(BaseEmbedder):
    """
    A QQMM-based embedder with the same public interface as `MMEmbed`.
    For the methods that accept (batch_size, max_length), we ignore them
    in practice because we embed inputs one-by-one.
    """

    def __init__(self, cuda_num: int, show_progress: bool = True):
        """
        :param cuda_num:  GPU device index; or pass 0 if you want to let the
                          'device_map' handle GPU usage automatically.
        :param show_progress: If True, display tqdm progress bars.
        """
        super().__init__()
        self.cuda_num = cuda_num
        self.show_progress = show_progress

        self.config = None
        self.processor = None
        self.model = None
        self.bot = None

        return

    def load_model(self):
        """
        Loads the QQMM model (and associated config + processor + bot) into memory.
        You can customize device mapping, dtype, etc.
        """
        print(">>> Loading QQMM config...")
        self.config = Parameters()
        self.config.merge_from_yaml(QQMM_CONFIG_YAML)

        print(">>> Building QQMM processor...")
        self.processor = build_processor(self.config.PROCESSOR_CONFIG, inferring=True)

        print(">>> Loading QQMM model...")
        # You can adjust torch_dtype and device_map to your preference.
        # If you want to strictly use `cuda:{self.cuda_num}`, remove device_map
        # or change it to a single device. E.g.:
        # device = f"cuda:{self.cuda_num}"
        # self.model = QQMMForCausalLM.from_pretrained(
        #     QQMM_MODEL_PATH, torch_dtype=torch.bfloat16, device_map={ "": device }
        # )
        self.model = QQMMForCausalLM.from_pretrained(
            QQMM_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",  # or device_map={"": f"cuda:{self.cuda_num}"}
        )

        print(">>> Building QQMM EmbedBot...")
        self.bot = EmbedBot(self.model, self.processor)

        print("Model loaded successfully.")

        return

    def delete_model(self):
        """
        Deletes model references for memory cleanup.
        """
        self.model = None
        self.processor = None
        self.bot = None
        print("Model deleted successfully.")

        return

    @torch.no_grad()
    def encode_queries(
        self,
        queries: List[str],
        output_filepath: Optional[str] = None,
        instruction: Union[str, Sequence[str]] = MODALITY_AGNOSTIC_QQMM_INSTRUCTION,
        batch_size: int = 4,  # ignored in this naive one-by-one approach
        max_length: int = 256 # unused by QQMM in this example
    ) -> torch.Tensor:
        """
        Encode a list of text queries into QQMM embeddings, one-by-one.
        
        :param queries: The text queries
        :param output_filepath: If given, save the resulting embeddings to that file
        :param instruction: Either a single string or a list of strings
                            (per-query override). If single, we prepend to each query.
        :return: torch.Tensor of shape (len(queries), QQMM_EMBED_DIM) on CPU
        """
        if self.bot is None:
            raise RuntimeError("Call `load_model()` before `encode_queries()`.")

        # Normalize instruction
        if isinstance(instruction, str):
            instructions = [instruction] * len(queries)
        else:
            if len(instruction) != len(queries):
                raise ValueError("`instruction` length must match `queries` length.")
            instructions = list(instruction)

        embeddings = []
        for i, (query, instr) in enumerate(zip(queries, instructions)):
            # We can unify into a single text: "Represent the given text for retrieval. " + ...
            # or incorporate the user-provided instruction. Up to you:
            # e.g. final_text = f"{instr}\n{query}"
            final_text = f"{instr}\n{query}"

            try:
                # QQMM returns shape [1, 3584] typically
                emb = self.bot.chat(text=final_text)
                emb = emb.squeeze(0).cpu()  # shape [3584]
            except Exception as e:
                print(f"❗ Encoding failed for query {i}: {e}. Using zeros.")
                emb = torch.zeros(QQMM_EMBED_DIM, dtype=torch.float32)

            embeddings.append(emb)

        # Combine into (N, 3584)
        out_tensor = torch.stack(embeddings, dim=0)

        if output_filepath:
            torch.save(out_tensor, output_filepath)
            print(f"💾 Saved embeddings → {output_filepath}  shape={out_tensor.shape}")

        return out_tensor

    @torch.no_grad()
    def encode_corpus(
        self,
        data: List[dict],
        out_embeddings_path: str,
        out_index_path: str,
        start_idx: Optional[int],
        end_idx: Optional[int],
        batch_size: int = 4,    # unused in the naive approach
        max_length: int = 256   # unused in the naive approach
    ) -> torch.Tensor:
        """
        Encode a list of items, each potentially containing text and/or images.
        We follow the same signature as `MMEmbed.encode_corpus`.

        data is a list of dict like:
            [
              {
                "id":  <some_id_str>,
                "target": {
                  "text":   <str>,
                  "images": [<paths to images>]
                }
              },
              ...
            ]

        We'll produce embeddings one by one. The text is embedded as text. If at
        least one image is valid, we will embed the first successful image. If
        both text and images exist, we create a combined approach or (optionally)
        embed text + image. You may define your own logic. Below we replicate
        your existing approach: if an image loads successfully, we pass it; else
        we pass only text.

        Returns the CPU embeddings of shape (N, 3584).
        """
        if self.bot is None:
            raise RuntimeError("Call `load_model()` before `encode_corpus()`.")

        # Prepare sub-slice
        total_len = len(data)
        if start_idx is not None:
            end_idx = min(end_idx, total_len)
            data_slice = data[start_idx:end_idx]
        else:
            data_slice = data

        idx_to_data_id = {}
        embeddings_list = []

        # For progress display
        rng = range(len(data_slice))
        if self.show_progress:
            from tqdm import tqdm
            rng = tqdm(rng, desc="Encoding corpus (QQMM)")

        for i in rng:
            # The 'i' is local index in this slice
            # Global index is (start_idx + i) if needed
            global_idx = (start_idx or 0) + i
            item = data_slice[i]

            data_id = item["id"]
            text_str = item["target"]["text"].strip()
            image_paths = item["target"]["images"]

            # Attempt to load the first valid image
            first_image = None
            for im_path in image_paths:
                try:
                    first_image = Image.open(im_path).convert("RGB")
                    break
                except Exception as e:
                    print(f"Warning: failed to load image at {im_path}: {e}")
                    continue

            # Decide how to embed: text or image or both
            # (For demonstration, we do: if we found an image, embed that alone,
            #  else embed the text.)
            if first_image is not None:
                # image-based embedding
                prompt = "Represent the given text and image for retrieval."
                try:
                    emb = self.bot.chat(text=prompt, image=[first_image])
                    emb = emb.squeeze(0).cpu()
                except Exception as e:
                    print(f"❗ Image embedding failed for {data_id}: {e}. Using zeros.")
                    emb = torch.zeros(QQMM_EMBED_DIM, dtype=torch.float32)
            else:
                # text-based embedding
                prompt = "Represent the given text for retrieval.\n" + text_str
                try:
                    emb = self.bot.chat(text=prompt)
                    emb = emb.squeeze(0).cpu()
                except Exception as e:
                    print(f"❗ Text embedding failed for {data_id}: {e}. Using zeros.")
                    emb = torch.zeros(QQMM_EMBED_DIM, dtype=torch.float32)

            embeddings_list.append(emb)
            idx_to_data_id[i] = data_id  # local index → data_id

        # Stack to a single tensor
        all_embeddings = torch.stack(embeddings_list, dim=0)

        # Build final output file name if slicing
        if start_idx is not None:
            base, ext = os.path.splitext(out_embeddings_path)
            final_emb_path = f"{base}_{start_idx}-{end_idx}{ext}"
        else:
            final_emb_path = out_embeddings_path

        torch.save(all_embeddings, final_emb_path)
        with open(out_index_path, "w") as f:
            json.dump(idx_to_data_id, f, indent=4)

        print(
            f"💾 Saved corpus embeddings to {final_emb_path} (shape={all_embeddings.shape})"
        )
        print(f"💾 Saved index to {out_index_path}")

        return all_embeddings



if __name__ == "__main__":


    sys.path.insert(1, '/root/omdr_workspace/algorithms/QQMM-embed')

    from qqmm.models import build_processor
    from qqmm.utils.parameter_manage import Parameters
    from qqmm.models.qqmm_nav_qwen2.modeling_qqmm import QQMMForCausalLM
    from qqmm.utils.chat import EmbedBot

    config = Parameters()
    config.merge_from_yaml('/root/omdr_workspace/algorithms/QQMM-embed/configs/embed/qqmm-embed/mmeb.yaml')

    print(">>> Building Model...")
    processor = build_processor(config.PROCESSOR_CONFIG, inferring=True)
    model = QQMMForCausalLM.from_pretrained('/mnt/sdc/jhyun/models/QQMM-embed-v1', torch_dtype=torch.bfloat16, device_map='cuda')
    bot = EmbedBot(model, processor)

    print(">>> Inference...")
    img = Image.open('/root/omdr_workspace/dog.jpeg').convert("RGB")
    img_feat = bot.chat(text='Represent the given image for retrieval.', image=[img])
    some_feat = bot.chat(text='Represent the given text for retrieval. Barack Obama is the 44th president of the United States. He was born in Hawaii and raised in Chicago. He is a member of the Democratic Party and served two terms from 2009 to 2017.')
    txt_feat = bot.chat(text='Retrieve an image that answers a question. NEVER RETRIEVE TEXT-ONLY evidences: Where is the first black president of the United States from?')


    print((img_feat).shape)
    print((txt_feat).shape)

    sim = (img_feat * txt_feat).sum()
    print('Similarity score: ', sim)

    sim = (some_feat * txt_feat).sum()
    print('Similarity score: ', sim)