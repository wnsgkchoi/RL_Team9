from __future__ import annotations

import argparse
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─────────────────────────────
# Model location
#   – Use HF hub by default; override with a local path if desired.
# ─────────────────────────────
QWEN_2_5_7B_PATH = "/mnt/sdc/jhyun/models/Qwen2.5-7B-Instruct"


class Qwen2_5_7B:
    """Batch-friendly wrapper around Qwen-2.5-7B-Instruct."""

    def __init__(
        self,
        model_path: str = QWEN_2_5_7B_PATH,
        *,
        device_map: str | Dict[str, int] = "auto",
        torch_dtype: str | torch.dtype = "auto",
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        # Tokenizer (includes chat template)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )

        # Causal-LM model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,        # "auto" → spread across GPUs if available
            torch_dtype=torch_dtype,       # "auto" → bfloat16 / fp16 where supported
            trust_remote_code=True,
        ).eval()                           # no grad needed

        self.system_prompt = system_prompt

        return

    # ───────────────────────────────────────────────────────────
    # Public inference API (same signature as the 72-B wrapper)
    # ───────────────────────────────────────────────────────────
    @torch.inference_mode()
    def infer_list(
        self,
        objects: List[Dict[str, Any]],
        *,
        max_tokens: int = 256,
        batch_size: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        do_sample: bool | None = None,
        show_progress: bool = False,
    ) -> List[str]:
        """
        Parameters
        ----------
        objects : List[Dict]
            Each dict must contain a `"text"` key with the user prompt.
        Returns
        -------
        List[str] – Generated responses, in the same order as the inputs.
        """
        if do_sample is None:                       # sensible default
            do_sample = temperature > 0

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

        results: List[str] = []

        # Mini-batch the work for speed / memory balance
        # for start in tqdm(
        #     range(0, len(objects), batch_size), desc="Inference Batches", unit="batch"
        # ):
        
        if show_progress:
            batch_iter = tqdm(
                range(0, len(objects), batch_size),
                desc="Inference Batches",
                unit="batch",
            )
        else:
            batch_iter = range(0, len(objects), batch_size)
        
        for start in batch_iter:
            batch_objs = objects[start : start + batch_size]

            # ── Build chat-formatted prompts ───────────────────────────────────
            prompts: List[str] = []
            for obj in batch_objs:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": obj["text"]},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)

            # ── Tokenize with padding & get per-sequence lengths ──────────────
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                return_length=True,      # gives us original prompt lengths
            ).to(self.model.device)

            lengths = enc.pop("length")  # <-- remove it so generate() sees no extra kwarg

            # ── Generate ─────────────────────────────────────────────────────
            generated = self.model.generate(**enc, **gen_kwargs)

            # ── Slice away the prompts & decode ──────────────────────────────
            for seq_idx, output_ids in enumerate(generated):
                prompt_len = int(lengths[seq_idx])          # use saved lengths
                reply_ids  = output_ids[prompt_len:]        # new tokens only
                results.append(
                    self.tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
                )

        return results








if __name__ == "__main__":
    qwen = Qwen2_5_7B()

    prompts = [
        {"text": "What is the capital of France?"},
        {"text": "What is the capital of Japan?"},
        {"text": "What is the capital of South Korea?"},
        {"text": "What is the capital of China?"},
        {"text": "What is the capital of the United States?"},
        {"text": "What is the capital of Canada?"},
        {"text": "What is the capital of Australia?"},
        {"text": "What is the capital of Germany?"},
        {"text": "What is the capital of Italy?"},
        {"text": "What is the capital of Spain?"},
    ] * 10  # 1 000 prompts total to stress-test batching

    outputs = qwen.infer_list(prompts, batch_size=32, max_tokens=50)

    for i, out in enumerate(outputs, 1):
        print(f"\n===== Output {i} =====\n{out}\n")

    pass