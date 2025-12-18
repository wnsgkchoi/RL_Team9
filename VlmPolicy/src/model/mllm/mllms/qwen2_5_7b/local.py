

from __future__ import annotations

from typing import List, Dict, Any, Tuple

import torch
import time
from tqdm import tqdm

from model.mllm.mllms.qwen2_5_7b._base import Qwen2_5_7B

from model.mllm.data_structure.input import GenerationInput, GenerationParameters
from model.mllm.data_structure.output import GenerationOutput, GenMetadata, TIME_MULTIPLIER_MS






class Qwen2_5_Local(Qwen2_5_7B):
    """Batch-friendly wrapper around Qwen-2.5-7B-Instruct."""

    
    def initiate(self):
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(self._local_path)
        
        """Dummy method for compatibility with other MLLM classes."""
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._local_path, 
            trust_remote_code   = True, 
            padding_side        = "left"
        )

        # Causal-LM model
        self._model = AutoModelForCausalLM.from_pretrained(
            self._local_path,
            device_map          = "auto",
            torch_dtype         = "auto",
            trust_remote_code   = True,
        ).eval()

        self._default_system_prompt = "You are a helpful assistant."
        
        self._default_parameters = GenerationParameters(
            show_progress       = True,
            batch_size          = 4,
            max_tokens          = 256,
            do_sample           = None,
            top_p               = 1.0,
            temperature         = 0.0,
            repetition_penalty  = 1.05
        )
            
        
        print()
        print("[Qwen2_5_Local] Model and tokenizer loaded successfully.")
        print()
        
        return


    @torch.inference_mode()
    def infer(
        self,
        gen_input:  GenerationInput,
        parameters: GenerationParameters = None
    ) -> Tuple[
        GenerationOutput,
        GenMetadata
    ]:
        """Single inference call. Delegates to infer_list for consistency."""
        gen_output_list, gen_metadata = self.infer_list([gen_input], parameters)
        return gen_output_list[0], gen_metadata



    # ───────────────────────────────────────────────────────────
    # Public inference API (same signature as the 72-B wrapper)
    # ───────────────────────────────────────────────────────────
    @torch.inference_mode()
    def infer_list(
        self,
        gen_inputs: List[GenerationInput],
        parameters: GenerationParameters = None,
    ) -> Tuple[
        List[GenerationOutput],
        GenMetadata
    ]:
        
        
        if not parameters:
            parameters = GenerationParameters()
        parameters += self._default_parameters
        
        show_progress       = parameters.show_progress
        batch_size          = parameters.batch_size
        max_tokens          = parameters.max_tokens
        do_sample           = parameters.do_sample
        top_p               = parameters.top_p
        temperature         = parameters.temperature
        repetition_penalty  = parameters.repetition_penalty
        
        metadata: GenMetadata = GenMetadata(
            time = 0,
            time_preprocess = 0,
            time_infer      = 0,
            time_postprocess= 0,
            num_inputs      = len(gen_inputs),
            batch_size      = batch_size,
            batch_iterations= 0,
        )
        
        # Preprocess
        pre_time_start = time.time()
        
        if do_sample is None:                       # sensible default
            do_sample = temperature > 0
            
        gen_kwargs = dict(
            max_new_tokens      = max_tokens,
            do_sample           = do_sample,
            temperature         = temperature,
            top_p               = top_p,
            repetition_penalty  = repetition_penalty,
        )
        
        gen_outputs: List[GenerationOutput] = []
        
        if show_progress:
            batch_iter = tqdm(
                range(0, len(gen_inputs), batch_size),
                desc = "Inference Batches",
                unit = "batch",
            )
        else:
            batch_iter = range(0, len(gen_inputs), batch_size)
            
        metadata.time_preprocess = (time.time() - pre_time_start) * TIME_MULTIPLIER_MS
        
        for start in batch_iter:
            
            
            # ── Build chat-formatted prompts ───────────────────────────────────
            
            inbatch_pretime_start = time.time()
            
            batch_inputs: List[GenerationInput] = gen_inputs[start : start + batch_size]

            prompts: List[str] = []
            for input in batch_inputs:
                
                self._warn_multimodal_inputs(input)
                messages = [
                    {"role": "system", "content": self._default_system_prompt if not input.system_prompt else input.system_prompt},
                    {"role": "user",   "content": input.text_prompt},
                ]
                prompt = self._tokenizer.apply_chat_template(
                    messages, 
                    tokenize                = False, 
                    add_generation_prompt   = True
                )
                prompts.append(prompt)

            # ── Tokenize with padding & get per-sequence lengths ──────────────
            enc = self._tokenizer(
                prompts,
                return_tensors  = "pt",
                padding         = True,
                return_length   = True
            ).to(self._model.device)

            lengths = enc.pop("length")
            
            metadata.time_preprocess += (time.time() - inbatch_pretime_start) * TIME_MULTIPLIER_MS
            
        
            # ── Generate ─────────────────────────────────────────────────────
            
            inbatch_infertime_start = time.time()
            
            batch_generations = self._model.generate(**enc, **gen_kwargs)
            
            metadata.time_infer += (time.time() - inbatch_infertime_start) * TIME_MULTIPLIER_MS

            # ── Slice away the prompts & decode ──────────────────────────────
            
            inbatch_posttime_start = time.time()
            
            for seq_idx, output_ids in enumerate(batch_generations):
                prompt_len = int(lengths[seq_idx])          # use saved lengths
                reply_ids  = output_ids[prompt_len:]        # new tokens only
                generated_text = self._tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

                gen_outputs.append(GenerationOutput(text = generated_text))
                
            metadata.time_postprocess += (time.time() - inbatch_posttime_start) * TIME_MULTIPLIER_MS
        
        metadata.time = (time.time() - pre_time_start) * TIME_MULTIPLIER_MS

        return gen_outputs, metadata
    
    
    
    
    
    
if __name__ == "__main__":

    qwen = Qwen2_5_Local()
    qwen.initiate()

    inputs = [
        GenerationInput(text_prompt = "What is the capital of France?",),
        GenerationInput(text_prompt = "What is the capital of Germany?",),
        GenerationInput(text_prompt = "What is the capital of Italy?",),
        GenerationInput(text_prompt = "What is the capital of Spain?",),
        GenerationInput(text_prompt = "What is the capital of Japan?",),
        GenerationInput(text_prompt = "What is the capital of South Korea?",),
        GenerationInput(text_prompt = "What is the capital of China?",),
        GenerationInput(text_prompt = "What is the capital of the United States?",),
        GenerationInput(text_prompt = "What is the capital of Canada?",),
        GenerationInput(text_prompt = "What is the capital of Australia?",),
    ] * 10

    outputs, metadata = qwen.infer_list(inputs)

    for i, out in enumerate(outputs, 1):
        print(f"\n===== Output {i} =====\n{out.text}\n")
    print(f"\n===== Metadata =====\n{metadata}\n")

    input = GenerationInput(text_prompt = "Explain the theory of relativity in simple terms.",)
    output, metadata = qwen.infer(input)
    print(f"\n===== Output =====\n{output.text}\n")
    print(f"\n===== Metadata =====\n{metadata}\n")

    # Image input test (note: qwen2_5_7b is text-only, images will be ignored with warning)
    import os
    cat_image_path = "/root/omdr_workspace/src/model/test_files/cat.png"
    if os.path.exists(cat_image_path):
        print("\n===== Image Input Test (Text-Only Model) =====")
        print("Note: This model does not support images. Image paths will be ignored.\n")
        input_with_image = GenerationInput(
            text_prompt = "What do you see in this image? Describe it in detail.",
            image_paths = [cat_image_path]
        )
        output_image, metadata_image = qwen.infer(input_with_image)
        print(f"\n===== Output (Image Input Ignored) =====\n{output_image.text}\n")
        print(f"Metadata: {metadata_image}\n")
    else:
        print(f"\n===== Image Test Skipped =====")
        print(f"Image not found at {cat_image_path}\n")

    pass