"""
A second, independent LLM call that reviews a finished puzzle.

It can only REJECT. It can never make an invalid puzzle valid: the strict
validators have already run before the critic is asked.

Two design choices, both learned from a real failure (a sentence about a game
"called off due to rain" that fit soccer, tennis AND hockey, and slipped through):

  * Every wrong option is judged on its own ("does THIS word also fit?"). A single
    broad question ("which options fit?") is answered "none" far too easily.
  * The reviewer can be a different, larger model than the one that wrote the puzzle,
    because a model shares its own blind spots.
"""

import json

import config
from ai import llm_client

# If the reviewer model turns out to be unavailable for this key we stop trying it
# for the rest of the session instead of failing on every puzzle.
_reviewer_model_broken = False


def reset():
    """Forget that the reviewer model was unavailable (used by tests)."""
    global _reviewer_model_broken

    _reviewer_model_broken = False


def _prompt(entry, theme):
    wrong = entry["options"][1:]
    template = ", ".join(f'"{w}": false' for w in wrong)

    return f"""You are a strict quality reviewer for a fill-in-the-blank word game.

Sentence: {entry['sentence']}
Correct answer: {entry['word']}
Wrong options: {", ".join(wrong)}

Task 1. Take the wrong options ONE AT A TIME. Put each into the blank and decide:
could a reasonable person consider the sentence still correct and sensible with that
word? If the sentence is vague and several words would fit, the answer is true.
Be strict - a good puzzle has exactly ONE word that fits.

Task 2. Does the correct answer clearly belong to the theme "{theme}"?

Return ONLY this JSON, nothing else. Each option is true if it ALSO fits the
sentence (a problem) and false if it clearly does not:
{{"options": {{{template}}}, "fits_theme": true, "issue": ""}}"""


def _parse(text):
    if not text:
        return None

    start = text.find("{")

    if start == -1:
        return None

    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, dict) else None


def _says_fits(value):
    """True for true / "yes" / {"fits": true} and similar spellings."""
    if isinstance(value, dict):
        value = value.get("fits", value.get("also_fits", False))

    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "fits", "y")

    return value is True


def _ambiguous_options(data, wrong):
    """Wrong options the reviewer says also fit (both reply formats accepted)."""
    found = []

    verdicts = data.get("options")

    if isinstance(verdicts, dict):
        for name, value in verdicts.items():
            if str(name).upper() in wrong and _says_fits(value):
                found.append(str(name).upper())

    for name in data.get("ambiguous_options") or []:      # the older reply format
        if str(name).upper() in wrong:
            found.append(str(name).upper())

    return sorted(set(found))


def _ask(prompt, llm):
    """Ask the reviewer model; fall back to the main model if it is unavailable."""
    global _reviewer_model_broken

    reviewer = config.CRITIC_MODEL

    if reviewer and not _reviewer_model_broken:
        text = llm(prompt, max_tokens=250, temperature=0.0,
                   timeout=config.LIVE_AI_TIMEOUT, model=reviewer)

        if text is not None:
            return text, reviewer

        _reviewer_model_broken = True

    text = llm(prompt, max_tokens=250, temperature=0.0, timeout=config.LIVE_AI_TIMEOUT)

    return text, config.API_MODEL


def review(entry, theme, llm=None):
    """
    Returns {"called", "reject", "reason", "model"}.
    An unavailable or unreadable critic never blocks a puzzle.
    """
    llm = llm or llm_client.generate_text

    text, model = _ask(_prompt(entry, theme), llm)

    data = _parse(text)

    if data is None:
        return {"called": text is not None, "reject": False,
                "reason": "critic unavailable", "model": model}

    wrong = {w.upper() for w in entry["options"][1:]}

    ambiguous = _ambiguous_options(data, wrong)

    if ambiguous:
        return {
            "called": True,
            "reject": True,
            "model": model,
            "reason": f"{', '.join(ambiguous)} could also fit the sentence - make the "
                      "sentence more specific (a clue only the answer satisfies) or "
                      "replace the ambiguous wrong option(s)",
        }

    if data.get("fits_theme") is False:
        return {
            "called": True,
            "reject": True,
            "model": model,
            "reason": f"the word does not clearly fit the theme '{theme}'",
        }

    return {"called": True, "reject": False, "reason": "critic approved", "model": model}
