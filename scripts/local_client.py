# local_client.py
# Replaced to talk to local vLLM (OpenAI-compatible) instead of OCI

import base64
import json as _json
import re
import mimetypes
import time
import threading
import os

import requests

# ===================== CONFIG =====================

# vLLM OpenAI-compatible endpoint
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

# Default model name served by vLLM (matches --served-model-name)
DEFAULT_VLM_MODEL = os.getenv("VLLM_MODEL_NAME", "llama-3_2-11b-vision")

# If you still want some global rate limiting across threads, set >0.0
MIN_INTERVAL = float(os.getenv("VLLM_MIN_INTERVAL", "0.0"))  # seconds

_rate_lock = threading.Lock()
_last_call_ts = 0.0


def _rate_limit():
    """Optional global rate limiter across threads (mainly for remote APIs)."""
    global _last_call_ts
    if MIN_INTERVAL <= 0:
        return
    with _rate_lock:
        now = time.time()
        wait = _last_call_ts + MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
            now = time.time()
        _last_call_ts = now


# ===================== Helpers =====================

def _encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _guess_mime(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        return "image/jpeg"
    return mime


def _post_chat(payload: dict) -> dict:
    """
    POST to local vLLM /v1/chat/completions and return parsed JSON.
    Raises requests.HTTPError on non-2xx.
    """
    url = f"{VLLM_BASE_URL}/chat/completions"

    _rate_limit()

    r = requests.post(url, json=payload, timeout=240)
    r.raise_for_status()
    return r.json()


# ===================== Vision call =====================

def call_vision_model(model_id: str, image_path: str, prompt: str) -> str:
    """
    Drop-in replacement for the old OCI vision call.

    Now:
    - Sends a single OpenAI-style /chat/completions request to local vLLM
    - Uses 'model_id' as the OpenAI model name if provided,
      otherwise falls back to DEFAULT_VLM_MODEL.
    """
    model_name = model_id or DEFAULT_VLM_MODEL

    mime = _guess_mime(image_path)
    b64 = _encode_image_b64(image_path)
    data_url = f"data:{mime};base64,{b64}"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        "max_tokens": 500,
        "temperature": 0.75,
        "top_p": 0.7,
        # top_k is not part of the OpenAI API; vLLM may ignore it anyway
    }

    try:
        resp_json = _post_chat(payload)
    except Exception as e:
        return f"__ERROR__: HTTP_error: {type(e).__name__}: {e}"

    try:
        # vLLM/OpenAI: content is typically a plain string
        return resp_json["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__ERROR__: parse_error: {type(e).__name__}: {e}; raw={resp_json!r}"


# ===================== Judge model call =====================

def call_judge_model(
    judge_model_id: str,
    system_prompt: str,
    user_content: str,
) -> dict:
    """
    Local judge using the same vLLM model (or a separate one if you serve it).

    Keeps the same signature as before, but now:
    - Uses OpenAI-style chat to local vLLM
    - Tries to extract a JSON object from the model's response (like before)
    """
    model_name = judge_model_id or DEFAULT_VLM_MODEL

    full_text = (
        system_prompt.strip()
        + "\n\n"
        + "=== INPUT TO JUDGE ===\n"
        + user_content.strip()
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_text},
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    try:
        resp_json = _post_chat(payload)
    except Exception as e:
        return {
            "parse_error": f"http_error: {type(e).__name__}: {e}",
            "raw": "",
        }

    try:
        content = resp_json["choices"][0]["message"]["content"]
        raw = content.strip() if isinstance(content, str) else _json.dumps(content)
    except Exception as e:
        return {
            "parse_error": f"no_text_in_response: {e}",
            "raw": _json.dumps(resp_json),
        }

    if not raw:
        return {
            "parse_error": "empty_response",
            "raw": _json.dumps(resp_json),
        }

    # Try to extract first {...} JSON block as before
    try:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {
                "parse_error": "no_json_found",
                "raw": raw,
            }
        json_str = match.group(0).strip()
        return _json.loads(json_str)
    except Exception as e:
        return {
            "parse_error": f"json_decode_error: {e}",
            "raw": raw,
        }
