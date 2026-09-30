import random
from dataclasses import replace

import config
from ai import llm_client
from agent.service import generate_with_agent
from ai.live_generator import generate_live_puzzle
from ai.local_generator import generate_local_puzzle
from ai.puzzle_bank import pick_puzzle


def _shuffled(puzzle, source):
    """Copy with shuffled options and a source label (never mutates the original)."""
    options = list(puzzle.options)
    random.shuffle(options)

    return replace(puzzle, options=options, source=puzzle.source or source)


def _from_ai(theme, difficulty, mode, used_words, use_agent=False):
    if config.PUBLIC_MODE:
        return None  # a public site never spends the AI allowance on puzzles, whatever else is configured

    if not llm_client.is_available():
        return None

    if use_agent:
        puzzle = generate_with_agent(theme, difficulty, used_words)

        if puzzle is not None:
            return puzzle

        # the agent could not finish - try the fast one-shot generator

    return generate_live_puzzle(theme, difficulty, used_words)


def _from_bank(theme, difficulty, mode, used_words, use_agent=False):
    length = config.WORD_LENGTH if mode == "Wordle" else None

    return pick_puzzle(theme, difficulty, used_words, length=length)


def _from_local(theme, difficulty, mode, used_words, use_agent=False):
    return generate_local_puzzle(theme, difficulty, mode)


SOURCES = {
    "ai": _from_ai,
    "bank": _from_bank,
    "local": _from_local,
}


def get_puzzle(theme, difficulty, mode, used_words=None, use_agent=False):
    """
    Returns a Puzzle (with .source set) or None.

    Sources are tried in config.PUZZLE_ORDER (default: ai -> bank -> local).
    Whenever one fails or returns nothing, the next one takes over.
    """
    used_words = used_words or []

    order = [n for n in config.PUZZLE_ORDER if n in SOURCES]

    if "local" not in order:
        order.append("local")  # always keep a last resort

    for name in order:
        try:
            puzzle = SOURCES[name](theme, difficulty, mode, used_words, use_agent)
        except Exception as exc:  # a broken source must never crash the game
            print(f"[puzzle_manager] source '{name}' failed: {exc}")
            continue

        if puzzle is None:
            continue

        # Wordle needs exactly WORD_LENGTH letters, whatever the source
        if mode == "Wordle" and len(puzzle.word) != config.WORD_LENGTH:
            continue

        return _shuffled(puzzle, name)

    return None
