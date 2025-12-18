from __future__ import annotations

import os
import base64
import mimetypes
import traceback
import time
from typing import List, Tuple, Dict, Any, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

from model.mllm.mllms.chatgpt5._base import ChatGPT5
from model.mllm.data_structure.input import GenerationInput, GenerationParameters
from model.mllm.data_structure.output import GenerationOutput, GenMetadata, APIUsage
import utils.directory as DirectoryUtils
import utils.file_io as FileIOUtils
from env import CHATGPT_API


# Default system prompt
DEFAULT_SYSTEM_PROMPT = "You are a careful vision+text assistant. Follow the user's image labels exactly."


class ChatGPT5_Client(ChatGPT5):
    """
    Client for ChatGPT-5 remote inference via OpenAI API.

    This class provides inference capabilities using OpenAI's Responses API
    for vision+text tasks.
    """

    def __init__(self) -> None:
        """Initialize the client.

        Loads configuration from config file and sets up OpenAI client.
        """
        super().__init__()

        # Load configuration
        config_dir = os.path.join(DirectoryUtils.get_mllm_config_dir(), "chatgpt5")
        config = FileIOUtils.read_yaml(os.path.join(config_dir, "_base.yaml"))

        self._model_variant     = config.get("model_variant", "gpt-5")
        self._timeout           = config.get("timeout", 300)
        self._reasoning_effort  = config.get("reasoning_effort", "low")
        self._pricing_table     = config.get("pricing_usd_per_mtok", {})

        # Initialize OpenAI client
        client_kwargs: Dict[str, Any] = {}
        if CHATGPT_API:
            client_kwargs["api_key"] = CHATGPT_API
        if self._timeout:
            client_kwargs["timeout"] = self._timeout

        self._client = OpenAI(**client_kwargs)

        return

    def initiate(self) -> dict:
        """Initialize the model and return model information.

        Returns:
            dict: Model information including model_name and status
        """
        print()
        print(f"[ChatGPT5_Client] Using model: {self._model_variant}")
        print(f"[ChatGPT5_Client] API client initialized successfully.")
        print()

        return {
            "model_name":       self._model_name,
            "model_variant":    self._model_variant,
            "status":           "initialized",
        }



    def infer(
        self,
        gen_input: GenerationInput,
        parameters: GenerationParameters = None,
    ) -> Tuple[
        GenerationOutput, 
        GenMetadata
    ]:
        
        """Single inference call via OpenAI API.

        Args:
            gen_input: Input for generation
            parameters: Optional generation parameters

        Returns:
            Tuple of (GenerationOutput, GenMetadata)
        """
        # Convert GenerationInput to ChatGPT format


        input_obj = {
            "text":     gen_input.text_prompt or "",
            "images":   gen_input.image_paths or [],
            "text_map": gen_input.text_map or "",
        }

        system_prompt = gen_input.system_prompt or DEFAULT_SYSTEM_PROMPT

        # Build input payload
        
        time_pre_start = time.perf_counter()
        input_payload = self._obj_to_inputformat(input_obj, system_prompt=system_prompt)
        time_pre = time.perf_counter() - time_pre_start

        # Extract reasoning effort from parameters, fallback to config default
        reasoning_effort = self._reasoning_effort
        if parameters is not None and parameters.api_parameters is not None:
            reasoning_effort = parameters.api_parameters.effort

        # Call API
        t0 = time.perf_counter()
        response = self._safe_generate(
            input_payload = input_payload,
            reasoning = {"effort": reasoning_effort},
        )
        time_infer = time.perf_counter() - t0

        # Extract results

        # Create output
        
        time_post_start = time.perf_counter()
        gen_output = GenerationOutput(text = self._extract_text(response))
        time_post = time.perf_counter() - time_post_start
        
        metadata = GenMetadata(
            time            = time_infer + time_pre + time_post,
            time_preprocess = time_pre,
            time_infer      = time_infer,
            time_postprocess= time_post,
            num_inputs      = 1,
            api_usage       = self._calculate_usage(response),
        )

        return gen_output, metadata



    def infer_list(
        self,
        gen_inputs: List[GenerationInput],
        parameters: GenerationParameters = None,
    ) -> Tuple[
        List[GenerationOutput],
        GenMetadata
    ]:
        
        """Batch inference call via OpenAI API.

        Args:
            gen_inputs: List of inputs for generation
            parameters: Optional generation parameters (not used for ChatGPT)

        Returns:
            Tuple of (List[GenerationOutput], GenMetadata)
        """
        
        gen_outputs = []
        total_time  = 0.0
        all_usage   = []

        # Process each input sequentially
        for gen_input in gen_inputs:
            gen_output, metadata = self.infer(gen_input, parameters)
            gen_outputs.append(gen_output)
            total_time += metadata.time or 0.0

            # Collect usage info if available
            if metadata.api_usage is not None:
                all_usage.append(metadata.api_usage)

        # Create combined metadata
        combined_metadata = GenMetadata(
            time                = total_time,
            time_infer          = total_time,
            num_inputs          = len(gen_inputs),
            batch_size          = 1,
            batch_iterations    = len(gen_inputs),
            api_usage           = self._aggregate_usage(all_usage) if all_usage else None,
        )

        return gen_outputs, combined_metadata

    # ──────────────────────────────────────────────────────────────────────
    # Helper methods (adapted from original chatgpt5.py)
    # ──────────────────────────────────────────────────────────────────────

    def _obj_to_inputformat(
        self, 
        obj: Dict, 
        system_prompt: Optional[str] = None
    ) -> List[Dict]:
        
        """
        Convert a simple dict to Responses API input payload format.

        Args:
            obj: Dict with "text" and "images" keys
            system_prompt: Optional system prompt

        Returns:
            List of role blocks for the API
        """
        
        raw_text: str = obj.get("text", "") or ""
        images: List[str] = obj.get("images", []) or []
        text_map: str = obj.get("text_map", "") or ""

        user_content: List[Dict[str, Any]] = []
        if raw_text:
            user_content.append({"type": "input_text", "text": raw_text})

        for idx, path in enumerate(images, start=1):
            # Convert local path to data URL
            url = path
            if not (isinstance(url, str) and (url.startswith("http://") or url.startswith("https://") or url.startswith("data:"))):
                try:
                    url = self._path_to_data_url(str(url))
                except Exception:
                    continue  # Skip unreadable images

            user_content.append({"type": "input_text", "text": f"[Image {idx}]"})
            user_content.append({"type": "input_text", "text": text_map})
            user_content.append({"type": "input_image", "image_url": url})

        payload: List[Dict[str, Any]] = []
        if system_prompt:
            payload.append({
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            })
        payload.append({
            "role": "user",
            "content": user_content if user_content else [{"type": "input_text", "text": ""}],
        })

        return payload


    def _safe_generate(
        self, 
        *, 
        input_payload: List[Dict], 
        reasoning: Optional[Dict[str, Any]] = None
    ):
        
        """Call OpenAI API and catch network/rate/API errors."""
        
        try:
            return self._client.responses.create(
                model = self._model_variant,
                input = input_payload,
                reasoning = reasoning or {"effort": "low"},
            )
        except (APIError, APIConnectionError, RateLimitError, APITimeoutError):
            print("[warn] OpenAI API error → returning empty answer")
            traceback.print_exc(limit=1)
            return None
        except Exception:
            print("[warn] Unexpected error → returning empty answer")
            traceback.print_exc(limit=1)
            return None


    def _extract_text(
        self, 
        response_obj
    ) -> str:
        
        """Get model's text output from response."""
        
        if response_obj is None:
            return ""
        text = getattr(response_obj, "output_text", None)
        if text is not None:
            return text
        try:
            parts = response_obj.output[0].content
            return "".join(getattr(p, "text", "") for p in parts if getattr(p, "type", "") == "output_text")
        except Exception:
            return ""


    def _calculate_usage(
        self, 
        response_obj
    ) -> Optional[APIUsage]:
        
        """Calculate usage and cost from response."""
        
        if response_obj is None or getattr(response_obj, "usage", None) is None:
            return None

        u = response_obj.usage
        in_tok = getattr(u, "input_tokens", 0)
        out_tok = getattr(u, "output_tokens", 0)

        # Some models report cached input tokens (discounted)
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0)

        # Determine pricing key from response.model
        model_name = getattr(response_obj, "model", None) or self._model_variant
        pricing_key = self._resolve_pricing_key(model_name)
        rates = self._pricing_table.get(pricing_key)

        # If we don't have a rate, return tokens only
        if not rates:
            return APIUsage(
                model                   = model_name,
                rate_info               = None,
                input_tokens            = in_tok,
                output_tokens           = out_tok,
                cached_input_tokens     = cached,
                estimated_cost_usd      = 0.0,
            )

        # Convert per 1M tokens to per token
        IN_RATE = rates["input"] / 1_000_000.0
        IN_CACHED = rates["cached_input"] / 1_000_000.0
        OUT_RATE = rates["output"] / 1_000_000.0

        paid_uncached = max(in_tok - cached, 0)
        usd = paid_uncached * IN_RATE + cached * IN_CACHED + out_tok * OUT_RATE

        return APIUsage(
            model                   = model_name,
            rate_info               = rates,
            input_tokens            = in_tok,
            output_tokens           = out_tok,
            cached_input_tokens     = cached,
            estimated_cost_usd      = usd,
        )


    def _resolve_pricing_key(
        self, 
        model_name: Optional[str]
    ) -> Optional[str]:
        
        """Map a concrete model name to pricing table key."""
        if not model_name:
            return None
        m = model_name.lower()
        if "nano" in m:
            return "gpt-5-nano"
        if "mini" in m:
            return "gpt-5-mini"
        return "gpt-5"


    def _aggregate_usage(
        self,
        usage_list: List[APIUsage]
    ) -> APIUsage:
        """Aggregate usage statistics from multiple calls."""
        if not usage_list:
            return None

        total_input = 0
        total_output = 0
        total_cached = 0
        total_cost = 0.0

        # Use first model name and rate_info (assuming all calls use same model)
        model_name = usage_list[0].model if usage_list else ""
        rate_info = usage_list[0].rate_info if usage_list else None

        for usage in usage_list:
            total_input     += usage.input_tokens
            total_output    += usage.output_tokens
            total_cached    += usage.cached_input_tokens
            total_cost      += usage.estimated_cost_usd

        return APIUsage(
            model                   = model_name,
            rate_info               = rate_info,
            input_tokens            = total_input,
            output_tokens           = total_output,
            cached_input_tokens     = total_cached,
            estimated_cost_usd      = total_cost,
        )


    @staticmethod
    def _path_to_data_url(path: str) -> str:
        """Encode a local image as a data URL."""
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "application/octet-stream"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"


if __name__ == "__main__":

    # Example usage
    client = ChatGPT5_Client()

    # Initialize
    model_info = client.initiate()
    print(f"Model info: {model_info}")

    # Single inference
    input1 = GenerationInput(text_prompt="What is the capital of France?")
    output1, metadata1 = client.infer(input1)
    print(f"\n===== Single Inference =====\n{output1.text}\n")
    print(f"Metadata: {metadata1}\n")

    # Batch inference
    inputs = [
        GenerationInput(text_prompt = "What is the capital of France?"),
        GenerationInput(text_prompt = "What is the capital of Germany?"),
        GenerationInput(text_prompt = "What is the capital of Italy?"),
    ]

    outputs, metadata = client.infer_list(inputs)

    for i, out in enumerate(outputs, 1):
        print(f"\n===== Output {i} =====\n{out.text}\n")
    print(f"\n===== Metadata =====\n{metadata}\n")

    # Image understanding test
    import os
    cat_image_path = "/root/omdr_workspace/src/model/test_files/cat.png"
    if os.path.exists(cat_image_path):
        input_image = GenerationInput(
            text_prompt = "What do you see in this image? Describe it in detail.",
            image_paths = [cat_image_path]
        )
        output_image, metadata_image = client.infer(input_image)
        print(f"\n===== Image Understanding Test =====\n{output_image.text}\n")
        print(f"Metadata: {metadata_image}\n")
        if metadata_image.api_usage:
            print(f"API Usage: {metadata_image.api_usage}\n")
    else:
        print(f"\n===== Image Understanding Test =====")
        print(f"Skipped: Image not found at {cat_image_path}\n")
