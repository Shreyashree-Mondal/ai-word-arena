"""
One small, provider-agnostic entry point for every AI feature.

    generate_text(prompt) -> str | None

It NEVER raises: if the provider is off, misconfigured, rate-limited or
offline it returns None, and every caller has a non-AI fallback.
"""

import functools
import importlib.util
import json
import os
import collections
import re
import threading
import time

import requests

import config


# ---------------------------------------
# Pacing: stay under the provider's tokens-per-minute limit
# ---------------------------------------

# Providers meter tokens PER MODEL, so each model gets its own budget window.
_usage = collections.defaultdict(collections.deque)   # model -> (time, estimated tokens)
_usage_lock = threading.Lock()


def reset_pacing():
    with _usage_lock:
        _usage.clear()


# What the provider actually reported for finished requests (not estimates).
_totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def usage_snapshot():
    with _usage_lock:
        return dict(_totals)


def usage_since(snapshot):
    """Tokens used since usage_snapshot() was taken (approximate if sessions overlap)."""
    with _usage_lock:
        diff = {k: _totals[k] - snapshot.get(k, 0) for k in _totals}

    diff["total_tokens"] = diff["prompt_tokens"] + diff["completion_tokens"]

    return diff


def _record_usage(usage):
    if not isinstance(usage, dict):
        return

    try:
        prompt = max(0, int(usage.get("prompt_tokens") or 0))
        completion = max(0, int(usage.get("completion_tokens") or 0))
    except (TypeError, ValueError):
        return

    with _usage_lock:
        _totals["calls"] += 1
        _totals["prompt_tokens"] += prompt
        _totals["completion_tokens"] += completion


def _estimate_tokens(prompt, reply_room):
    """Prompt tokens (about 4 characters each) plus the reply room the request reserves."""
    return len(prompt) // 4 + reply_room


def _wait_for_budget(estimate, model=None):
    """
    Sleep just long enough that this request fits in the last 60 seconds' budget.
    Never waits longer than API_PACING_MAX_WAIT in total: after that the request is
    sent anyway and the 429 handling (or the caller's fallback) takes over.
    """
    limit = config.API_TPM_LIMIT

    if not limit:
        return

    waited = 0.0

    while True:
        now = time.monotonic()

        with _usage_lock:
            window = _usage[model]

            while window and now - window[0][0] >= 60:
                window.popleft()

            used = sum(tokens for _, tokens in window)

            if not window or used + estimate <= limit or waited >= config.API_PACING_MAX_WAIT:
                window.append((now, estimate))
                return

            pause = 60 - (now - window[0][0]) + 0.05

        pause = min(pause, config.API_PACING_MAX_WAIT - waited)

        time.sleep(pause)

        waited += pause


def _api_key():
    return os.getenv("AIWORD_API_KEY") or os.getenv("GROQ_API_KEY")


def is_available():
    """True if a generation call is worth attempting."""
    provider = config.AI_PROVIDER

    if provider == "api":
        return bool(_api_key())

    if provider == "local":
        return importlib.util.find_spec("transformers") is not None

    return False


def _strip_thinking(text):
    """Remove <think>...</think> blocks some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------
# Hosted API (OpenAI-compatible: Groq, Gemini, OpenRouter, ...)
# ---------------------------------------

def _is_reasoning_model(model):
    name = (model or "").lower()

    return "gpt-oss" in name or "qwen3" in name or "-r1" in name or "reasoning" in name


def _extra_tokens(model=None):
    setting = str(config.API_EXTRA_TOKENS).strip().lower()

    if setting == "auto":
        return 500 if _is_reasoning_model(model or config.API_MODEL) else 0

    try:
        return max(0, int(setting))
    except ValueError:
        return 0


def _reasoning_effort(model=None):
    setting = str(config.API_REASONING_EFFORT).strip().lower()

    if setting == "auto":
        return "low" if "gpt-oss" in (model or config.API_MODEL or "").lower() else None

    return None if setting in ("", "off", "none") else setting


def _error_detail(response):
    """Groq explains what is wrong in the response body - keep that text."""
    try:
        error = response.json().get("error", {})
        detail = error.get("message") if isinstance(error, dict) else str(error)
    except (ValueError, AttributeError):
        detail = ""

    return (detail or response.text or "")[:300].replace("\n", " ")


def _salvage_failed_generation(response):
    """
    Some models answer with their built-in tool-call format. When no tools were
    declared the server refuses with 400 "tool_use_failed" but includes what the
    model tried to say. That text is exactly the action we asked for, so use it.
    """
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        return None

    if isinstance(error, dict):
        text = error.get("failed_generation")

        if isinstance(text, str) and text.strip():
            return text

    return None


def _is_bad_generation(response):
    """
    True when the server refused because the MODEL produced unusable output
    (as opposed to a problem with our request, our key or the service itself).
    """
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        return False

    if not isinstance(error, dict):
        return False

    message = str(error.get("message", "")).lower()

    return (
        error.get("code") in ("tool_use_failed", "json_validate_failed")
        or "failed_generation" in message
        or "could not be parsed" in message
        or "failed to call a function" in message
    )


def _tool_call_as_action(message):
    """A tool_calls entry -> the JSON action text the agent understands."""
    calls = message.get("tool_calls") or []

    if not calls or not isinstance(calls[0], dict):
        return ""

    function = calls[0].get("function") or {}

    return json.dumps(
        {"name": function.get("name"), "arguments": function.get("arguments")}
    )


_WAIT_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}


def _retry_wait(response):
    """Seconds the service asks us to wait after a rate limit, or None if unknown."""
    header = (getattr(response, "headers", None) or {}).get("retry-after")

    try:
        if header is not None:
            return float(header)
    except (TypeError, ValueError):
        pass

    match = re.search(r"try again in\s+([0-9hms.\s]+)", _error_detail(response))

    if not match:
        return None

    parts = _WAIT_PART.findall(match.group(1))

    return sum(float(n) * _UNIT_SECONDS[u] for n, u in parts) if parts else None


def _call_api(prompt, max_tokens, temperature, timeout=None, model=None):
    key = _api_key()
    model = model or config.API_MODEL

    if not key:
        return None

    reply_room = max_tokens + _extra_tokens(model)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": reply_room,
        "temperature": temperature,
    }

    effort = _reasoning_effort(model)

    if effort:
        payload["reasoning_effort"] = effort

    def send():
        return requests.post(
            config.API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout or config.AI_TIMEOUT_SECONDS,
        )

    _wait_for_budget(_estimate_tokens(prompt, reply_room), model)

    response = send()

    if response.status_code == 429:
        wait = _retry_wait(response)

        if wait is not None and wait <= config.API_RETRY_MAX_WAIT:
            time.sleep(wait + 0.25)   # one polite retry, never a loop
            response = send()

    if response.status_code == 400:
        salvaged = _salvage_failed_generation(response)

        if salvaged:
            return salvaged

        if _is_bad_generation(response):
            # The model garbled its reply. That is not an outage: hand back an empty
            # reply so the caller can treat it as one bad answer and carry on.
            print(f"[llm_client] the model produced unusable output: {_error_detail(response)}")
            return ""

    if not response.ok:
        raise RuntimeError(f"{response.status_code} - {_error_detail(response)}")

    data = response.json()

    _record_usage(data.get("usage"))

    message = data["choices"][0]["message"]

    text = _strip_thinking(message.get("content") or "")

    return text or _tool_call_as_action(message)


# ---------------------------------------
# Local Hugging Face model (loaded lazily, only if selected)
# ---------------------------------------

@functools.lru_cache(maxsize=1)
def _load_local_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[llm_client] Loading local model {config.LOCAL_MODEL_NAME} ...")

    tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        config.LOCAL_MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    print("[llm_client] Model device:", model.device)

    return tokenizer, model


def _call_local(prompt, max_tokens, temperature):
    tokenizer, model = _load_local_model()

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode ONLY the newly generated tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return _strip_thinking(
        tokenizer.decode(new_tokens, skip_special_tokens=True)
    )


# ---------------------------------------
# Public API
# ---------------------------------------

def generate_text(prompt, max_tokens=200, temperature=0.7, timeout=None, model=None):
    provider = config.AI_PROVIDER

    try:
        if provider == "api":
            return _call_api(prompt, max_tokens, temperature, timeout, model)

        if provider == "local":
            return _call_local(prompt, max_tokens, temperature)

    except Exception as exc:  # network, rate limit, missing GPU, bad JSON...
        print(f"[llm_client] {provider} generation failed: {exc}")

    return None
