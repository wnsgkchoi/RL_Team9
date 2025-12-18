
import os
import json
import re
import argparse
from typing import List, Union, Sequence, Dict

from tqdm import tqdm
from PIL import Image


import torch
import torch.nn.functional as F
from embedders.base_embedder import BaseEmbedder



MODALITY_AGNOSTIC_MMEMBED_RETRIEVAL_INSTRUCTION = "Retrieve passage, table or image (maybe with a caption) that provides an answer to the given query."
MMEMBED_PATH = "/mnt/sdc/jhyun/models/MM-Embed"



def parse_arguments():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--corpus_in_filepath',     type = str, required = False,       help = 'Path to input .tsv file')
    parser.add_argument('--out_embeddings_path',    type = str, required = False,       help = 'Path to save output .pt file')
    parser.add_argument('--out_index_path',         type = str, required = False,       help = 'Path to save output .pt file')
    parser.add_argument('--batch_size',             type = int, default  = 4,           help = 'Batch size for encoding')
    parser.add_argument('--max_length',             type = int, default  = 128,         help = 'CUDA device number')
    parser.add_argument('--offset',                 type = int, default  = 0,           help = 'Starting offset of the rows to process')
    parser.add_argument('--size',                   type = int, default  = 10000,       help = 'Number of rows to process')
    
    return parser.parse_args()



def main():
    
    args = parse_arguments()
    
    with open(args.corpus_in_filepath, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
        
    mmembed = MMEmbed(0)
    
    mmembed.encode_corpus(
        data                = corpus_data,
        out_embeddings_path = args.out_embeddings_path,
        out_index_path      = args.out_index_path,
        start_idx           = args.offset,
        end_idx             = args.offset + args.size
    )
    
    return












class MMEmbed(BaseEmbedder):
    
    def __init__(
        self, 
        cuda_num, 
        show_progress = True
    ) -> None:
        
        self.mode = None
        self.model_path = MMEMBED_PATH
        self.cuda_num = cuda_num
        self.show_progress = show_progress
        
        return



    def load_model(self) -> None:

        from transformers import AutoModel
        
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code = True)
        self.model.to('cuda:' + str(self.cuda_num))
        print("Model loaded successfully.")

        return
    
    
    
    def delete_model(self) -> None:
        self.model = None
        print("Model deleted successfully.")
        
        return




    # ------------------------------------------------------------------
    #  Text-only queries — one-by-one (no batching)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_queries(
        self,
        queries:            List[str],
        output_filepath:    Union[str, None]            = None,
        instruction:        Union[str, Sequence[str]]   = MODALITY_AGNOSTIC_MMEMBED_RETRIEVAL_INSTRUCTION,
        batch_size:         int                         = 4,
        max_length:         int                         = 256,
    ) -> torch.Tensor:
        
        """
        Encode each query separately with MM-Embed.

        • `instruction` may be a single str (applied to every query) or a list
        of equal length.  Behaviour matches the old batched version.
        • Returns a CPU tensor of shape (len(queries) × 4096).
        • If `output_filepath` is provided, saves with torch.save().
        """
        
        if self.model is None:
            raise RuntimeError("Call `load_model()` before `encode_queries()`")

        if instruction == None:
            instruction = MODALITY_AGNOSTIC_MMEMBED_RETRIEVAL_INSTRUCTION

        # ── normalise instruction list ─────────────────────────────────
        if isinstance(instruction, str):
            instr_list = [instruction] * len(queries)
        else:
            if len(instruction) != len(queries):
                raise ValueError("`instruction` length must match `queries` length")
            instr_list = list(instruction)

        embed_dim = 4096
        embeddings: List[torch.Tensor] = []

        iterator = range(len(queries))
        if self.show_progress:
            iterator = tqdm(iterator, desc="EncodeQ-one-by-one")

        # ── encode each query on its own ───────────────────────────────
        for i in iterator:
            q      = queries[i]
            instr  = instr_list[i]

            # MM-Embed expects list[dict]; we give exactly one element.
            inp = [{"txt": q}]

            try:
                out   = self.model.encode(
                    inp,
                    is_query=True,
                    instruction=instr,
                    max_length=max_length,
                )
                emb = out["hidden_states"].squeeze(0).cpu()
            except Exception as e:
                print(f"❗ Encoding failed for query {i}: {e} – using zeros")
                emb = torch.zeros(embed_dim, dtype=torch.float32)

            embeddings.append(emb)

        tensor = torch.stack(embeddings, dim=0)

        # ── optional save ──────────────────────────────────────────────
        if output_filepath:
            torch.save(tensor, output_filepath)
            print(f"💾 Saved embeddings → {output_filepath}  shape={tensor.shape}")

        return tensor



    def encode_corpus(self, 
        data:                   dict[str, str], 
        out_embeddings_path:    str,
        out_index_path:         str,
        start_idx:              int,
        end_idx:                int,
        batch_size:             int = 4,
        max_length:             int = 256
    ) -> torch.Tensor:

        # Input: {
        #   ["id": id_1, "target": str_1], ...
        # }

        instruction = ""

        all_inputs = []
        idx_to_data_id = {}
        print("Loading data...")
        for i, data_obj in enumerate(tqdm(data)):
            
            data_id     = data_obj['id']
            data_txt    = data_obj['target']["text"]
            image_paths = data_obj["target"]["image_paths"]
            
            clean_text = data_txt.strip()

            image = None
            success = False
            
            for image_path in image_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    success = True
                except Exception as e:
                    print(f"Warning: failed to load image at {image_path}: {e}")
                if success:
                    break

            input_obj = {'txt': clean_text}
            if image is not None: input_obj['img'] = image
            all_inputs.append(input_obj)
            
            idx_to_data_id[i] = data_id
        
        if start_idx != None:
            end_idx = min(len(all_inputs), end_idx)
            all_inputs = all_inputs[start_idx : end_idx]
        else:
            all_inputs = all_inputs

        # Batched encoding for corpus
        all_embeddings = []
        if self.show_progress:
            iterating = tqdm(range(0, len(all_inputs), batch_size))
        else:
            iterating = range(0, len(all_inputs), batch_size)
        for i in iterating:
            batch_inputs = all_inputs[i : i + batch_size]
            with torch.no_grad():
                try:
                    outputs = self.model.encode(batch_inputs, instruction = instruction, max_length = max_length)
                    embeddings = outputs['hidden_states']
                except Exception as e:
                    print(f"Error during model encoding: {e}")
                    embeddings = torch.zeros((len(batch_inputs), 4096), dtype=torch.float32).to('cuda:0')
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim = 0)
        
        # split `out_embeddings_path` by extension and base name
        # Add `start_idx-end_idx` to end of the base name
        if start_idx != None:
            final_embeddings_path = os.path.splitext(out_embeddings_path)[0] + f"_{start_idx}-{end_idx}" + os.path.splitext(out_embeddings_path)[1]
        else:
            final_embeddings_path = out_embeddings_path
        
        torch.save(all_embeddings, final_embeddings_path)
        with open(out_index_path, "w") as f:
            json.dump(idx_to_data_id, f, indent = 4)

        return all_embeddings












if __name__ == '__main__':
    main()

