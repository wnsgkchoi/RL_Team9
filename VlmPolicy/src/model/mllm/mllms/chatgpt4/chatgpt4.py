# chatgpt41_vl.py
import base64, mimetypes, traceback, time, json
from typing import List, Dict, Optional, Any, Union, Tuple

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
from env import chatgpt_api  # 환경변수/별도 모듈로 키 주입 권장

DEFAULT_SYSTEM_PROMPT = "You are a careful vision+text assistant. Follow the user's image labels exactly."

# ──────────────────────────────────────────────────────────────────────────────
# Pricing tables (USD per 1M tokens)
#   참고 출처: OpenAI 공식 페이지
#   - GPT-4.1 계열 (입력/캐시입력/출력): gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
#   - GPT-4o 계열 (입력/캐시입력/출력): gpt-4o, gpt-4o-mini
# ──────────────────────────────────────────────────────────────────────────────
PRICING_USD_PER_MTOK_41 = {
    "gpt-4.1":       {"input": 3.00,  "cached_input": 0.75,  "output": 12.00},
    "gpt-4.1-mini":  {"input": 0.80,  "cached_input": 0.20,  "output": 3.20},
    "gpt-4.1-nano":  {"input": 0.20,  "cached_input": 0.05,  "output": 0.80},
}

PRICING_USD_PER_MTOK_4O = {
    # GPT-4o 텍스트/비전
    "gpt-4o":        {"input": 2.50,  "cached_input": 0.00,  "output": 10.00},
    # GPT-4o mini
    "gpt-4o-mini":   {"input": 0.15,  "cached_input": 0.08,  "output": 0.60},
}

def _path_to_data_url(path: str) -> str:
    """Encode a local image as a data URL suitable for input_image/image_url."""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

class ChatGPT41:
    """
    Single-request wrapper around OpenAI Responses API for GPT-4.1 / GPT-4o families.
    - Supports: gpt-4.1 / gpt-4.1-mini / gpt-4.1-nano / gpt-4o / gpt-4o-mini
    - Returns (text, metadata) where metadata includes usage + response_time
    - Cost is estimated from usage via the pricing tables above
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,  # seconds
    ):
        self.model = self._normalize_model_name(model)
        # 두 테이블을 합쳐서 사용
        self.pricing_table: Dict[str, Dict[str, float]] = {**PRICING_USD_PER_MTOK_41, **PRICING_USD_PER_MTOK_4O}

        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if timeout:
            client_kwargs["timeout"] = timeout
        self.client = OpenAI(**client_kwargs)

    # public API
    def infer(
        self,
        input_object: Dict,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        reasoning: Optional[Dict[str, Any]] = None,  # 유지하되 내부에서 무시
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build final payload, call Responses API once, return (text, metadata).
        metadata = {"usage": <dict|None>, "response_time": <float seconds>}
        """
        input_payload = self._obj_to_inputformat(input_object, system_prompt=system_prompt)

        t0 = time.perf_counter()
        response = self._safe_generate(input_payload=input_payload)  # reasoning 전달 X
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
            if isinstance(p, dict) and p.get("url"):
                url = p["url"]
            elif isinstance(p, str):
                url = p
            else:
                continue

            if not (isinstance(url, str) and (url.startswith("http://") or url.startswith("https://") or url.startswith("data:"))):
                try:
                    url = _path_to_data_url(str(url))
                except Exception:
                    continue

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

    def _safe_generate(self, *, input_payload: List[Dict]):
        """Call OpenAI API and catch network/rate/API errors. (No 'reasoning' arg)"""
        try:
            return self.client.responses.create(
                model=self.model,
                input=input_payload,
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
        """Get model's text output (Responses API preferred)."""
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

    # ───────────────────────────────
    # Usage & Cost
    # ───────────────────────────────
    def _normalize_model_name(self, name: str) -> str:
        """Map minor aliases to canonical names (e.g., 'chatgpt-4.1*' → 'gpt-4.1*', 'chatgpt-4o*' → 'gpt-4o*')."""
        m = (name or "").strip()
        low = m.lower()
        if low.startswith("chatgpt-4.1"):
            return "gpt-" + m.split("-", 1)[1]
        if low.startswith("chatgpt-4o"):
            return "gpt-" + m.split("-", 1)[1]
        return m

    def _resolve_pricing_key(self, model_name: Optional[str]) -> Optional[str]:
        """
        Map a concrete model name to pricing table keys.
        Examples:
          "gpt-4o-2024-08-06"  → "gpt-4o"
          "gpt-4o-mini"        → "gpt-4o-mini"
          "gpt-4.1-mini-2025"  → "gpt-4.1-mini"
        """
        if not model_name:
            return None
        m = model_name.lower()

        # GPT-4o family
        if "gpt-4o" in m:
            return "gpt-4o-mini" if "mini" in m else "gpt-4o"

        # GPT-4.1 family
        if "gpt-4.1" in m:
            if "nano" in m:
                return "gpt-4.1-nano"
            if "mini" in m:
                return "gpt-4.1-mini"
            return "gpt-4.1"

        return None

    def _calculate_usage(self, response_obj) -> Optional[Dict[str, Any]]:
        if response_obj is None or getattr(response_obj, "usage", None) is None:
            return None

        u = response_obj.usage
        in_tok  = getattr(u, "input_tokens", 0)
        out_tok = getattr(u, "output_tokens", 0)
        tot_tok = getattr(u, "total_tokens", in_tok + out_tok)
        cached  = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0)

        model_name = getattr(response_obj, "model", None) or self.model
        pricing_key = self._resolve_pricing_key(model_name)
        rates = self.pricing_table.get(pricing_key)

        if not rates:
            return {
                "model_reported": model_name,
                "pricing_key": pricing_key,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": tot_tok,
                "cached_input_tokens": cached,
                "estimated_cost_usd": None,
                "estimated_cost_usd_str": None,
                "note": "No pricing found for this model key; tokens only.",
            }

        IN_RATE   = rates["input"]        / 1_000_000.0
        IN_CACHED = rates["cached_input"] / 1_000_000.0
        OUT_RATE  = rates["output"]       / 1_000_000.0

        paid_uncached = max(in_tok - cached, 0)
        usd = paid_uncached * IN_RATE + cached * IN_CACHED + out_tok * OUT_RATE

        return {
            "model_reported": model_name,
            "pricing_key": pricing_key,
            "rates_per_mtok_usd": rates,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
            "cached_input_tokens": cached,
            "estimated_cost_usd": usd,
            "estimated_cost_usd_str": f"${usd:.6f}",
        }

# ──────────────────────────────────────────────────────────────────────
# quick sanity-check
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ⚠️ 예시: 별도 모듈/환경변수에서 API 키 주입
    gpt = ChatGPT41(
        model="gpt-4o",           # gpt-4o / gpt-4o-mini / gpt-4.1 / gpt-4.1-mini / gpt-4.1-nano
        api_key=chatgpt_api,
        timeout=30,
    )

    input_obj = {
        "text": "Say 'OK'",
        "images": []  # ["./local.png", ...]도 지원(자동 data: URL 변환)
    }

    out_text, metadata = gpt.infer(
        input_object=input_obj,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        reasoning=None,  # 받아도 내부에서 전달하지 않음
    )

    print()
    print("🪄 Generated answer:\n", out_text)
    print()
    print("🔎 Metadata:")
    print(json.dumps(metadata, indent=2))
    print()
