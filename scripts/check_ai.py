"""
Check that the AI connection works, and explain what is wrong if it does not.

    set GROQ_API_KEY=your-key          (Windows)   /   export GROQ_API_KEY=your-key
    python scripts/check_ai.py            # is the connection working?
    python scripts/check_ai.py --limits   # also show each model's rate limits

It never prints your key.
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

import config  # noqa: E402
from ai import llm_client  # noqa: E402

# Models that cannot chat (speech, audio, safety classifiers, embeddings)
NOT_FOR_CHAT = ("whisper", "tts", "orpheus", "guard", "safeguard", "embed", "playai")


def models_url():
    return config.API_URL.rsplit("/chat/completions", 1)[0] + "/models"


def limits_for(model):
    """One tiny request to a model, reading the rate-limit headers the service sends back."""
    try:
        response = requests.post(
            config.API_URL,
            headers={"Authorization": f"Bearer {llm_client._api_key()}"},
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
            timeout=30,
        )
    except requests.RequestException as exc:
        return {"error": type(exc).__name__}

    headers = response.headers or {}

    info = {
        "tokens_per_minute": headers.get("x-ratelimit-limit-tokens"),
        "requests_per_day": headers.get("x-ratelimit-limit-requests"),
        "tokens_left": headers.get("x-ratelimit-remaining-tokens"),
    }

    if not response.ok:
        info["error"] = f"{response.status_code} {llm_client._error_detail(response)}"[:120]

    return info


def show_limits(models, out):
    out("\nRate limits of your key (measured with one tiny request per model):")

    measured = {}

    for model in sorted(models):
        info = limits_for(model)

        if "error" in info and info.get("tokens_per_minute") is None:
            out(f"  {model:<30} could not be measured: {info['error']}")
            continue

        per_minute = info.get("tokens_per_minute") or "not reported"
        per_day = info.get("requests_per_day") or "not reported"
        left = info.get("tokens_left")

        out(f"  {model:<30} {per_minute:>12} tokens/min   {per_day:>12} requests/day"
            + (f"   ({left} tokens left this minute)" if left else ""))

        try:
            measured[model] = int(per_minute)
        except ValueError:
            pass

    if measured:
        top = max(measured.values())
        leaders = sorted(m for m, v in measured.items() if v == top)

        if len(leaders) == 1:
            out(f"\nHighest tokens-per-minute limit: {leaders[0]} ({top}).")
        else:
            out(f"\nHighest tokens-per-minute limit: {top}, shared equally by {', '.join(leaders)}.")

        out("The agent re-sends its conversation on every step, so a higher limit means more puzzles per minute.")


def main(argv=(), out=print):
    wants_limits = "--limits" in list(argv)

    out(f"Provider : {config.AI_PROVIDER}")
    out(f"Address  : {config.API_URL}")
    out(f"Model    : {config.API_MODEL}")

    if config.AI_PROVIDER != "api":
        out("\nThe hosted API is not selected. Set GROQ_API_KEY (or AIWORD_API_KEY) first.")
        return 1

    if not llm_client.is_available():
        out("\nNo API key found in this window. Set it first, e.g. (Windows):")
        out("  set GROQ_API_KEY=your-key")
        return 1

    out("Key      : set")

    # 1. What can this key use?
    chat_models = []

    try:
        response = requests.get(
            models_url(),
            headers={"Authorization": f"Bearer {llm_client._api_key()}"},
            timeout=20,
        )

        if response.ok:
            ids = [m.get("id", "") for m in response.json().get("data", [])]
            chat_models = [i for i in ids if not any(w in i.lower() for w in NOT_FOR_CHAT)]

            out(f"\nModels your key can use for chat ({len(chat_models)}):")

            for name in sorted(chat_models):
                out(f"  - {name}")

            if wants_limits:
                show_limits(chat_models, out)

            if config.CRITIC_MODEL and config.CRITIC_MODEL not in ids:
                out(f"\nNote: the reviewer model '{config.CRITIC_MODEL}' is not available to your key,")
                out("so puzzle reviews will use the main model instead (works, but is a weaker check).")
                out('To choose another reviewer: set "AIWORD_CRITIC_MODEL=<model name>"')

            if config.API_MODEL not in ids:
                out(f"\n!! The configured model '{config.API_MODEL}' is NOT in that list.")
                out("   Pick one from the list above, then (Windows):")
                out('   set "AIWORD_API_MODEL=<model name>"')
                return 2
        elif response.status_code in (401, 403):
            out(f"\n!! The key was rejected ({response.status_code}). Check that it was copied completely.")
            return 2
        else:
            out(f"\n!! The model list request returned {response.status_code}. Check the address above.")
            return 2
    except requests.RequestException as exc:
        out(f"\n!! Could not reach the service: {type(exc).__name__}. Check your internet connection.")
        return 2

    # 2. Does an actual request work?
    out("\nSending a test request...")

    reply = llm_client.generate_text("Reply with the single word: ready", max_tokens=20)

    if reply is None:
        out("!! The test request failed (see the message above for the reason).")
        return 3

    out(f"Reply    : {reply.strip()[:80]!r}")
    out("\nAll good - the AI connection works.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
