#!/usr/bin/env python3
# llama3_1_70b.py
# Author: Joohyung Yun – DSLab, POSTECH
#
# Batch wrapper for Meta-Llama-3.1-70B-Instruct served by an sglang HTTP
# endpoint.  API-compatible with llama3_1_8b.py.

from __future__ import annotations

from typing import List, Dict, Any
import json
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────
DEFAULT_SGLANG_ADDR = "http://localhost:30000/generate"   # ← 70B server
LLAMA_HF_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# ─────────────────────────────────────────────────────────────


class Llama3_1_70B:
    """
    >>> llama = Llama3_1_70B()
    >>> outs  = llama.infer([{"text": "hello"}])
    """

    def __init__(
        self,
        model_path: str | None = None,              # kept for parity; ignored
        *,
        device_map:         str | Dict[str, int]    = "auto",
        torch_dtype:        str | torch.dtype       = "auto",
        system_prompt:      str                     = "You are a helpful assistant.",
        llm_addr:           str                     = DEFAULT_SGLANG_ADDR,
        timeout:            float | None            = None,
        use_chat_template:  bool                    = True,
    ) -> None:
        self.llm_addr = llm_addr
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_HF_ID, trust_remote_code=True, padding_side="left"
        )

        # Provide a fallback template if HF tokenizer lacks one
        if (
            use_chat_template
            and not getattr(self.tokenizer, "chat_template", None)
        ):
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if loop.index0 == 0 %}<|begin_of_text|>{% endif %}"
                "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n"
                "{{ message['content'] }}<|eot_id|>\n"
                "{% endfor %}"
                "<|start_header_id|>assistant<|end_header_id|>"
            )

    # ───────────────────────── infer ─────────────────────────
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
        Same signature / semantics as llama3_1_8b.Llama3_1_8B.infer().
        """
        sampling_params: Dict[str, Any] = {"max_new_tokens": max_tokens}
        if temperature != 0.0:
            sampling_params["temperature"] = temperature
        if top_p != 1.0:
            sampling_params["top_p"] = top_p
        if repetition_penalty != 1.05:
            sampling_params["repetition_penalty"] = repetition_penalty
        if do_sample is True:
            sampling_params["do_sample"] = True

        results: List[str] = []
        batch_iter = (
            tqdm(range(0, len(objects), batch_size),
                 desc="Inference Batches", unit="batch")
            if show_progress else range(0, len(objects), batch_size)
        )

        for start in batch_iter:
            batch = objects[start:start + batch_size]

            # Build chat-templated prompts
            prompts: List[str] = []
            for obj in batch:
                if self.use_chat_template:
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": obj["text"]},
                    ]
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt = (
                        f"{self.system_prompt}\n\nUser: {obj['text']}\nAssistant:"
                    )
                prompts.append(prompt)

            try:
                resp = requests.post(
                    self.llm_addr,
                    json={"text": prompts, "sampling_params": sampling_params},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                payload = resp.json()  # list of dicts
            except Exception as e:
                try:
                    print("⚠️  sglang error payload:",
                          json.dumps(resp.json(), indent=2)[:800])
                except Exception:
                    pass
                raise RuntimeError(
                    f"sglang request failed ({e}). Check server logs."
                ) from e

            for item in payload:
                results.append(item.get("text", "").strip())

        return results


# ───────────────────── smoke-test ─────────────────────
if __name__ == "__main__":
    llama = Llama3_1_70B()

    demo_prompts = [
        {"text": "What is the capital of France?"},
        {"text": "Explain quantum entanglement in simple terms."},
        {"text": "Translate 'machine learning' into Korean."},
        {"text": "Suggest three weekend project ideas."},
    ] * 5  # 20 queries

    outputs = llama.infer_list(
        demo_prompts,
        batch_size=4,
        max_tokens=96,
        show_progress=True,
    )

    for i, out in enumerate(outputs, 1):
        print(f"\n===== Output {i} =====\n{out}\n")
