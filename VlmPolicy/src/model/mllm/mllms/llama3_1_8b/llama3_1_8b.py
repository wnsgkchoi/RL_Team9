#!/usr/bin/env python3
# llama3_1_8b.py
# Author: Joohyung Yun – DSLab, POSTECH
#
# Batch wrapper for Meta-Llama-3.1-8B-Instruct served by an sglang HTTP
# endpoint.  Mirrors the qwen2_5_7b.py API.

from __future__ import annotations

from typing import List, Dict, Any
import json
import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────
DEFAULT_SGLANG_ADDR = "http://localhost:30000/generate"
LLAMA_HF_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# ─────────────────────────────────────────────────────────────


class Llama3_1_8B:
    
    def __init__(
        self,
        model_path: str | None = None,              # ignored (kept for parity)
        *,
        device_map: str | Dict[str, int] = "auto",  # ignored
        torch_dtype: str | torch.dtype = "auto",    # ignored
        system_prompt: str = "You are a helpful assistant.",
        llm_addr: str = DEFAULT_SGLANG_ADDR,
        timeout: float | None = None,
        use_chat_template: bool = True,
    ) -> None:
        
        self.llm_addr = llm_addr
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_HF_ID, trust_remote_code=True, padding_side="left"
        )

        # Provide a fallback template if missing
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
            
        return

    # ─────────────────────────────────────────────────────────
    # Public inference API
    # ─────────────────────────────────────────────────────────
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
        Accepts the same kwargs as Qwen2_5_7B.infer().
        """
        # Build *minimal* sampling params (expand only if non-default)
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

            # Compose prompt strings
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
                else:  # raw prompt (system + user)
                    prompt = (
                        f"{self.system_prompt}\n\nUser: {obj['text']}\nAssistant:"
                    )
                prompts.append(prompt)

            try:
                resp = requests.post(
                    self.llm_addr,
                    json = {"text": prompts, "sampling_params": sampling_params},
                    timeout = self.timeout,
                )
                resp.raise_for_status()
                payload = resp.json()  # list of dicts
            except Exception as e:
                # Show server error body to help debugging, then re-raise
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


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    llama = Llama3_1_8B()

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









# #!/usr/bin/env python3
# # llama3_1_8b.py
# # Author: Joohyung Yun – DSLab, POSTECH
# #
# # Batch-friendly wrapper around Meta-Llama-3.1-8B-Instruct that mirrors
# # the qwen2_5_7b.py API while following the official pipeline usage.

# from __future__ import annotations

# from typing import List, Dict, Any

# import torch
# from tqdm import tqdm
# from transformers import pipeline


# # Either a local folder or the HF repo ID
# LLAMA3_1_8B_PATH = (
#     "/mnt/sdc/jhyun/models/Llama-3.1-8B-Instruct"
#     # "meta-llama/Meta-Llama-3.1-8B-Instruct"  # ← use this to pull from Hub
# )


# class Llama3_1_8B:
#     """
#     >>> llama = Llama3_1_8B()
#     >>> outs  = llama.infer([{"text": "Hello"}])
#     """

#     def __init__(
#         self,
#         model_path: str = LLAMA3_1_8B_PATH,
#         *,
#         device_map: str | Dict[str, int] = "auto",
#         torch_dtype: str | torch.dtype = "auto",
#         system_prompt: str = "You are a helpful assistant.",
#     ) -> None:
#         # Map "auto" → bfloat16 to match the example snippet
#         dtype = (
#             torch.bfloat16
#             if torch_dtype in ("auto", "bfloat16", torch.bfloat16)
#             else torch_dtype
#         )

#         self.pipe = pipeline(
#             "text-generation",
#             model=model_path,
#             model_kwargs={"torch_dtype": dtype},
#             device_map=device_map,
#         )
#         self.system_prompt = system_prompt

#     # ───────────────────────── infer ─────────────────────────
#     @torch.inference_mode()
#     def infer(
#         self,
#         objects: List[Dict[str, Any]],
#         *,
#         max_tokens: int = 256,
#         batch_size: int = 4,
#         temperature: float = 0.0,
#         top_p: float = 1.0,
#         repetition_penalty: float = 1.05,
#         do_sample: bool | None = None,
#         show_progress: bool = False,
#     ) -> List[str]:
#         if do_sample is None:
#             do_sample = temperature > 0

#         gen_kwargs = dict(
#             max_new_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             repetition_penalty=repetition_penalty,
#             do_sample=do_sample,
#             # pipeline auto-sets pad_token_id; no need to pass it
#         )

#         results: List[str] = []
#         batch_iter = (
#             tqdm(
#                 range(0, len(objects), batch_size),
#                 desc="Inference Batches",
#                 unit="batch",
#             )
#             if show_progress
#             else range(0, len(objects), batch_size)
#         )

#         for start in batch_iter:
#             batch_objs = objects[start : start + batch_size]

#             # Build batch of chat-style message lists
#             batch_messages = [
#                 [
#                     {"role": "system", "content": self.system_prompt},
#                     {"role": "user", "content": obj["text"]},
#                 ]
#                 for obj in batch_objs
#             ]

#             batch_out = self.pipe(batch_messages, **gen_kwargs)

#             # The pipeline can return several shapes; normalise them.
#             for out in batch_out:
#                 reply = self._extract_assistant_reply(out)
#                 results.append(reply)

#         return results

#     # ─────────────────────────────────────────────────────────
#     # Helpers
#     # ─────────────────────────────────────────────────────────
#     @staticmethod
#     def _extract_assistant_reply(out_obj) -> str:
#         """
#         Robustly pull the assistant's text from any pipeline return form:
#         1. {'generated_text': '...'}  (older versions)
#         2. list[{'role': 'system', ...}, ..., {'role': 'assistant', 'content': '...'}]
#         3. plain string
#         """
#         # case 1
#         if isinstance(out_obj, dict) and "generated_text" in out_obj:
#             maybe = out_obj["generated_text"]
#             # This may still be list-of-messages (HF ≥ 4.41); fall through.
#             out_obj = maybe

#         # case 2
#         if isinstance(out_obj, list):
#             # assume last message is assistant
#             last = out_obj[-1]
#             if isinstance(last, dict) and "content" in last:
#                 return str(last["content"]).strip()
#             return str(last).strip()

#         # case 3: already a string
#         return str(out_obj).strip()


# # ───────────────────── smoke-test ─────────────────────
# if __name__ == "__main__":
#     llama = Llama3_1_8B()

#     prompts = [
#         {"text": "What is the capital of USA?"},
#         {"text": "Explain quantum entanglement in simple terms."},
#         {"text": "What is machine learning?"},
#         {"text": "What should I do today?"},
#     ] * 5  # 20 queries

#     outputs = llama.infer(
#         prompts,
#         batch_size=4,
#         max_tokens=256,
#         show_progress=True,
#     )

#     for i, out in enumerate(outputs, 1):
#         print(f"\n===== Output {i} =====\n{out}\n")

