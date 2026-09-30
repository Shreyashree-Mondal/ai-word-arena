"""
Generates puzzles live with the AI, inside the game.

Every puzzle is validated exactly like bank puzzles (real 6-letter word,
no answer leak, 3 distractors). Returns None on ANY failure so the puzzle
manager can fall back to the bank.

To keep the game snappy and to respect free-tier rate limits, one AI call
asks for a small batch; the spares are kept in memory for the next rounds.
"""

import threading

import config
from ai import llm_client
from ai.bank_builder import build_prompt, parse_entries, validate_entry
from ai.puzzle_bank import append_entry, entry_to_puzzle

_lock = threading.Lock()

# (theme, difficulty) -> list of validated spare puzzle dicts
_buffer = {}


def clear_buffer():
    with _lock:
        _buffer.clear()


def _serve(entry):
    return entry_to_puzzle(entry)


def generate_live_puzzle(
    theme,
    difficulty,
    used_words=None,
    llm=None,
    bank_path=None,
):
    used = {w.upper() for w in (used_words or [])}

    key = (theme, difficulty)

    # 1) Spare puzzles from an earlier AI call (instant, no API request)
    with _lock:
        spares = _buffer.get(key, [])

        while spares:
            entry = spares.pop(0)

            if entry["word"] not in used:
                return _serve(entry)

    # 2) Ask the AI
    llm = llm or llm_client.generate_text

    for _ in range(config.LIVE_MAX_ATTEMPTS):
        text = llm(
            build_prompt(theme, difficulty, config.LIVE_BATCH_SIZE, used),
            max_tokens=900,
            temperature=0.8,
            timeout=config.LIVE_AI_TIMEOUT,
        )

        if text is None:
            return None  # provider down / rate limited -> caller uses the bank

        valid = []
        taken = set(used)

        for raw in parse_entries(text):
            entry, _reason = validate_entry(raw, theme, difficulty, taken)

            if entry is not None:
                valid.append(entry)
                taken.add(entry["word"])

        if not valid:
            continue  # nothing usable this time - try once more

        if config.AUTO_SAVE_AI_PUZZLES:
            for entry in valid:
                try:
                    append_entry(entry, bank_path)
                except OSError as exc:
                    print(f"[live_generator] could not save to bank: {exc}")

        first, rest = valid[0], valid[1:]

        with _lock:
            _buffer.setdefault(key, []).extend(rest)

        return _serve(first)

    return None
