# chatgpt5_vl.py
import base64, mimetypes, traceback, time, json
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
from env import CHATGPT_API


DEFAULT_SYSTEM_PROMPT = "You are a careful vision+text assistant. Follow the user's image labels exactly."

# ──────────────────────────────────────────────────────────────────────────────
# Pricing table (USD per 1M tokens)
# ──────────────────────────────────────────────────────────────────────────────
PRICING_USD_PER_MTOK = {
    "gpt-5":       {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini":  {"input": 0.25, "cached_input": 0.025, "output": 2.00 },
    "gpt-5-nano":  {"input": 0.05, "cached_input": 0.005, "output": 0.40 },
}


class ChatGPT5:
    """
    vLLM-free, single-request wrapper for OpenAI Responses API (vision+text).
    - Supports gpt-5 / gpt-5-mini / gpt-5-nano
    - Optional system_prompt per request
    - Returns (text, metadata) where metadata includes usage + response_time
    - Automatically estimates cost from usage using the pricing table above
    """

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: Optional[str] = CHATGPT_API,
        timeout: Optional[float] = None,  # seconds
    ):
        self.model = model
        self.pricing_table = PRICING_USD_PER_MTOK  # allow override if needed

        # OpenAI client (reads OPENAI_API_KEY by default)
        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if timeout:
            client_kwargs["timeout"] = timeout
        self.client = OpenAI(**client_kwargs)




    # ------------------------------------------------------------------ #
    # public API                                                          #
    # ------------------------------------------------------------------ #
    def infer(
        self,
        input_object: Dict,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        reasoning: Dict[str, Any] = {"effort": "low"},
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build final payload from input_object (and optional system_prompt),
        call Responses API once, return (text, metadata).
        metadata = {"usage": <dict|None>, "response_time": <float seconds>}
        On error, returns ("", {"usage": None, "response_time": elapsed}).
        """
        input_payload = self._obj_to_inputformat(input_object, system_prompt=system_prompt)

        t0 = time.perf_counter()
        response = self._safe_generate(
            input_payload = input_payload,
            reasoning = reasoning,
        )
        elapsed = time.perf_counter() - t0

        text = self._extract_text(response)
        usage = self._calculate_usage(response)

        metadata = {
            "usage": usage,
            "response_time": elapsed,
        }
        return text, metadata

    def _obj_to_inputformat(self, obj: Dict, system_prompt: Optional[str] = None) -> List[Dict]:
        """
        Convert a simple dict:
          {
            "text":   <user text prompt str>,
            "images": [<local path or http(s)/data URL> ...]
          }
        into a Responses API `input` payload (list of role blocks).
        """
        raw_text: str = obj.get("text", "") or ""
        images: List[Union[str, Dict[str, str]]] = obj.get("images", []) or []

        user_content: List[Dict[str, Any]] = []
        if raw_text:
            user_content.append({"type": "input_text", "text": raw_text})

        for idx, p in enumerate(images, start=1):
            # Resolve URL (support dict{"url":...} or string path/URL)
            if isinstance(p, dict) and p.get("url"):
                url = p["url"]
            elif isinstance(p, str):
                url = p
            else:
                continue

            # Convert local path → data URL
            if not (isinstance(url, str) and (url.startswith("http://") or url.startswith("https://") or url.startswith("data:"))):
                try:
                    url = _path_to_data_url(str(url))
                except Exception:
                    continue  # unreadable → skip

            user_content.append({"type": "input_text", "text": f"[Image {idx}]"})
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

    def _safe_generate(self, *, input_payload: List[Dict], reasoning: Optional[Dict[str, Any]] = None):
        """Call OpenAI API and catch network/rate/API errors."""
        try:
            return self.client.responses.create(
                model=self.model,
                input=input_payload,
                reasoning=reasoning or {"effort": "low"},
            )
        except (APIError, APIConnectionError, RateLimitError, APITimeoutError):
            print("[warn] OpenAI API error → returning empty answer")
            traceback.print_exc(limit=1)
            return None
        except Exception:
            print("[warn] Unexpected error → returning empty answer")
            traceback.print_exc(limit=1)
            return None

    def _extract_text(self, response_obj) -> str:
        """Get model’s text output (Responses API preferred)."""
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

    # ──────────────────────────────────────────────────────────────────
    # Usage & Cost
    # ──────────────────────────────────────────────────────────────────
    
    def _calculate_usage(self, response_obj) -> Optional[Dict[str, Any]]:
        if response_obj is None or getattr(response_obj, "usage", None) is None:
            return None

        u = response_obj.usage
        in_tok  = getattr(u, "input_tokens", 0)
        out_tok = getattr(u, "output_tokens", 0)
        tot_tok = getattr(u, "total_tokens", in_tok + out_tok)

        # Some models report cached input tokens (discounted)
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0)

        # Determine pricing key from response.model (fallback to self.model)
        model_name = getattr(response_obj, "model", None) or self.model
        pricing_key = self._resolve_pricing_key(model_name)
        rates = self.pricing_table.get(pricing_key)

        # If we don't have a rate, return tokens only.
        if not rates:
            return {
                "model_reported": model_name,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": tot_tok,
                "cached_input_tokens": cached,
                "estimated_cost_usd": None,
                "estimated_cost_usd_str": None,
                "note": "No pricing found for this model key; tokens only.",
            }

        # Convert per 1M tokens → per token
        IN_RATE      = rates["input"]        / 1_000_000.0
        IN_CACHED    = rates["cached_input"] / 1_000_000.0
        OUT_RATE     = rates["output"]       / 1_000_000.0

        paid_uncached = max(in_tok - cached, 0)
        usd = paid_uncached * IN_RATE + cached * IN_CACHED + out_tok * OUT_RATE

        return {
            "model": model_name,
            "pricing_key": pricing_key,
            "rates_per_mtok_usd": rates,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
            "cached_input_tokens": cached,
            "estimated_cost_usd": usd,
            "estimated_cost_usd_str": f"${usd:.6f}",
        }
    
    def _resolve_pricing_key(self, model_name: Optional[str]) -> Optional[str]:
        """
        Map a concrete model name to one of the keys in PRICING table.
        e.g., "gpt-5-mini-2025-08-01" → "gpt-5-mini"
        """
        if not model_name:
            return None
        m = model_name.lower()
        if "nano" in m:
            return "gpt-5-nano"
        if "mini" in m:
            return "gpt-5-mini"
        # Default bucket
        return "gpt-5"

    


def _path_to_data_url(path: str) -> str:
    """Encode a local image as a data URL suitable for input_image/image_url."""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"




@dataclass
class LLMUsage:
    
    calls:          int     = 0
    input_tokens:   int     = 0
    output_tokens:  int     = 0
    cost_usd:       float   = 0.0

    def increment_with_metadata(self, md: Optional[Dict[str, Any]]):
        if not md or not md.get("usage"):
            return
        self.calls += 1
        u = md["usage"]
        self.input_tokens  += int  (u.get("input_tokens", 0) or 0)
        self.output_tokens += int  (u.get("output_tokens", 0) or 0)
        self.cost_usd      += float(u.get("estimated_cost_usd", 0.0) or 0.0)
























# ──────────────────────────────────────────────────────────────────────
# quick sanity-check
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ⚠️ 권장: 키는 환경변수 OPENAI_API_KEY 로만 주입 (하드코딩 금지)
    # pip install -U openai
    gpt = ChatGPT5(
        model = "gpt-5",
        api_key = CHATGPT_API,
        timeout = 30,
    )
    
    # 1) 주어진 로컬 이미지 경로
    image_paths = [
        "/root/omdr_workspace/src/algorithms/sns/examples/bear1.png",
        "/root/omdr_workspace/src/algorithms/sns/examples/bear2.png",
        "/root/omdr_workspace/src/algorithms/sns/examples/bear3.png",
    ]

    # 2) 입력 객체(라벨 설명은 text, 이미지들은 로컬 경로)
    input_obj = {
        "text": (
            "You'll see three images. Treat them as Image 1..3 in order. "
            "When I say 'the third picture', I mean Image 3.\n\n"
            "Task: What is the color of the bear in Image 3?"
        ),
        "images": image_paths,
    }
    # input_obj = {
    #     "text": "Say 'OK'",
    #     "images": []
    # }


    # 3) system_prompt
    system_prompt = (
        "You are a careful vision+text assistant. "
        "Follow the user's image labels exactly."
    )

    # 4) 호출
    out_text, metadata = gpt.infer(
        input_object = input_obj,
        system_prompt = system_prompt,
        reasoning = {"effort": "low"},
    )

    print()
    print("🪄 Generated answer:\n", out_text)
    print()
    print("🔎 Metadata:")
    print(json.dumps(metadata, indent=4))
    print()
    
    
    