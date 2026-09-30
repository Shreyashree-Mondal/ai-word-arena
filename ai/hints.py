"""Extra-hint feature: AI clue when available, simple letter clue otherwise."""

import re

import config
from ai import llm_client

MAX_HINTS = config.MAX_HINTS


def fallback_hint(puzzle, level):
    """Non-AI hint that gets progressively more helpful."""
    word = puzzle.word.upper()

    if level <= 1:
        return f"The word has {len(word)} letters and starts with '{word[0]}'."

    if level == 2:
        return f"It ends with the letter '{word[-1]}'."

    return (
        f"It starts with '{word[:2]}' and ends with '{word[-2:]}'."
    )


def _build_prompt(puzzle, previous_hints):
    known = [puzzle.hint or puzzle.sentence] + list(previous_hints)

    known_text = "\n".join(f"- {c}" for c in known)

    return f"""You are helping a player in a word game.

Secret word: {puzzle.word}
Theme: {puzzle.theme}
Difficulty: {puzzle.difficulty}

Clues the player already has:
{known_text}

Write ONE new clue that helps them guess the secret word.

Rules:
- Exactly one sentence, under 25 words.
- Give different information from the clues above.
- NEVER write the secret word itself.
- Do not spell the word or give its letters.
- Return only the clue, nothing else."""


def _clean(text):
    if not text:
        return ""

    line = next((l.strip() for l in text.splitlines() if l.strip()), "")

    line = re.sub(r"^(hint|clue)\s*\d*\s*[:\-]\s*", "", line, flags=re.IGNORECASE)

    return line.strip(" \"'*-")


def _is_acceptable(clue, puzzle, previous_hints):
    if not clue or len(clue) > 220:
        return False

    if puzzle.word.lower() in clue.lower():
        return False

    seen = {h.lower() for h in previous_hints}

    return clue.lower() not in seen


def get_extra_hint(puzzle, previous_hints=None, use_ai=True, on_call=None):
    """
    Returns (hint_text, from_ai).
    Always returns a usable hint - the AI is a bonus, never a requirement.
    """
    previous_hints = list(previous_hints or [])

    if use_ai and llm_client.is_available():
        if on_call:
            on_call()   # counts the request itself, whether or not the reply turns out usable

        raw = llm_client.generate_text(
            _build_prompt(puzzle, previous_hints),
            max_tokens=80,
            temperature=0.7,
        )

        clue = _clean(raw)

        if _is_acceptable(clue, puzzle, previous_hints):
            return clue, True

    return fallback_hint(puzzle, len(previous_hints) + 1), False
