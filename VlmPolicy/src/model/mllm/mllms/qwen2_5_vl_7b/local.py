

from __future__ import annotations

from typing import List, Dict, Any, Tuple
import math

import torch
import time
import traceback
from tqdm import tqdm

from model.mllm.mllms.qwen2_5_vl_7b._base import Qwen2_5_VL_7B

from model.mllm.data_structure.input import GenerationInput, GenerationParameters
from model.mllm.data_structure.output import GenerationOutput, GenMetadata, TIME_MULTIPLIER_MS




class Qwen2_5_VL_Local(Qwen2_5_VL_7B):
    """Local inference wrapper for Qwen2.5-VL-7B-Instruct with vision support."""


    def initiate(self):

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        print(self._local_path)

        """Initialize the model, processor, and tokenizer."""
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._local_path,
            torch_dtype     = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map      = "auto",
        )

        self._processor = AutoProcessor.from_pretrained(self._local_path)

        # Choose one "main" device for small tensors
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" or not torch.cuda.is_available():
            self._tensor_device = torch.device("cpu")
        else:
            first_cuda = next(
                (d for d in self._model.hf_device_map.values() if "cuda" in str(d)),
                "cuda:0",
            )
            self._tensor_device = torch.device(first_cuda)

        self._process_vision_info = process_vision_info

        self._default_system_prompt = "You are a helpful assistant."

        self._default_parameters = GenerationParameters(
            show_progress       = True,
            batch_size          = 1,  # VL models typically use batch_size=1
            max_tokens          = 256,
            do_sample           = None,
            top_p               = 0.001,
            temperature         = 0.1,
            repetition_penalty  = 1.05
        )

        # Limits for multimedia inputs
        self._limit_mm_per_prompt = self._model_cfg["limit_mm_per_prompt"]


        print()
        print("[Qwen2_5_VL_Local] Model and processor loaded successfully.")
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
    # Public inference API
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
            time                = 0,
            time_preprocess     = 0,
            time_infer          = 0,
            time_postprocess    = 0,
            num_inputs          = len(gen_inputs),
            batch_size          = batch_size,
            batch_iterations    = 0,
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
            input_iter = tqdm(
                gen_inputs,
                desc = "Inference",
                unit = "input",
            )
        else:
            input_iter = gen_inputs

        metadata.time_preprocess = (time.time() - pre_time_start) * TIME_MULTIPLIER_MS

        # Process each input individually (VL models typically don't batch well)
        for gen_input in input_iter:

            inbatch_pretime_start = time.time()

            # Build messages in Qwen chat format
            messages = self._build_messages(gen_input)

            # Apply chat template
            text_prompt = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process vision info
            try:
                img_inputs, vid_inputs = self._process_vision_info(messages)
            except OSError as e:
                print("[OS ERROR] Truncated or corrupted image!")
                print(f"Input: {gen_input}")
                print(e)
                gen_outputs.append(GenerationOutput(text="[Error: Truncated Image]"))
                metadata.time_preprocess += (time.time() - inbatch_pretime_start) * TIME_MULTIPLIER_MS
                continue

            # Prepare inputs
            inputs = self._processor(
                text=[text_prompt],
                images=img_inputs,
                videos=vid_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._tensor_device)

            metadata.time_preprocess += (time.time() - inbatch_pretime_start) * TIME_MULTIPLIER_MS

            # Generate
            inbatch_infertime_start = time.time()

            out_ids = self._safe_generate(inputs, **gen_kwargs)

            metadata.time_infer += (time.time() - inbatch_infertime_start) * TIME_MULTIPLIER_MS

            # Decode
            inbatch_posttime_start = time.time()

            if out_ids is None:  # OOM
                gen_outputs.append(GenerationOutput(text=""))
            else:
                new_tokens = out_ids[0, inputs.input_ids.shape[1]:]
                generated_text = self._processor.decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                gen_outputs.append(GenerationOutput(text=generated_text))

            metadata.time_postprocess += (time.time() - inbatch_posttime_start) * TIME_MULTIPLIER_MS

        metadata.time = (time.time() - pre_time_start) * TIME_MULTIPLIER_MS
        metadata.batch_iterations = len(gen_inputs)

        return gen_outputs, metadata


    def _build_messages(self, gen_input: GenerationInput) -> List[Dict]:
        """
        Build Qwen chat messages from GenerationInput.
        Supports text, images, and videos.
        """
        # Limit multimedia inputs
        images = (gen_input.image_paths or [])[: self._limit_mm_per_prompt.get("image", math.inf)]
        videos = (gen_input.video_paths or [])[: self._limit_mm_per_prompt.get("video", math.inf)]
        text = gen_input.text_prompt or ""

        # Build content list
        content = []

        # Add images
        for img_path in images:
            content.append({"type": "image", "image": img_path})

        # Add videos
        for vid_path in videos:
            content.append({"type": "video", "video": vid_path})

        # Add text
        if text:
            content.append({"type": "text", "text": text})

        # Build messages
        messages = []

        # Add system prompt if provided
        system_prompt = gen_input.system_prompt or self._default_system_prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        # Add user message with content
        messages.append({
            "role": "user",
            "content": content if content else [{"type": "text", "text": ""}]
        })

        return messages


    def _safe_generate(self, inputs, **gen_kwargs):
        """
        Call self._model.generate() and catch CUDA OOM.
        On success  → tensor
        On OOM      → None
        """
        try:
            return self._model.generate(**inputs, **gen_kwargs)

        except torch.cuda.OutOfMemoryError:
            pass
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise

        # ---- OOM path ----------------------------------------------------
        print("[warn] CUDA OOM during generation → returning empty answer")
        traceback.print_exc(limit=1)
        torch.cuda.empty_cache()
        return None






if __name__ == "__main__":

    qwen = Qwen2_5_VL_Local()
    qwen.initiate()

    # Test with cat image
    cat_image_path = "/root/omdr_workspace/src/model/test_files/cat.png"

    inputs = [
        GenerationInput(
            text_prompt = "What do you see in this image? Describe it in detail.",
            image_paths = [cat_image_path]
        ),
        GenerationInput(
            text_prompt = "What is the capital of France?"
        ),
    ]

    outputs, metadata = qwen.infer_list(inputs)

    for i, out in enumerate(outputs, 1):
        print(f"\n===== Output {i} =====\n{out.text}\n")
    print(f"\n===== Metadata =====\n{metadata}\n")

    # Single inference test
    input_single = GenerationInput(
        text_prompt = "What animal is in this image?",
        image_paths = [cat_image_path]
    )
    output, metadata = qwen.infer(input_single)
    print(f"\n===== Single Output =====\n{output.text}\n")
    print(f"\n===== Metadata =====\n{metadata}\n")

    pass
