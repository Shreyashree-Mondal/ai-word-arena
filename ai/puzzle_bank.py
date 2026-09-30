"""Reads puzzles from data/puzzles.json (the offline puzzle bank)."""

import json
import os
import random
import threading
from pathlib import Path

from database.models import Puzzle

BANK_PATH = Path(__file__).resolve().parent.parent / "data" / "puzzles.json"

_write_lock = threading.Lock()


def load_entries(path=None):
    """Raw list of puzzle dicts. Missing or broken file -> empty list."""
    path = Path(path) if path else BANK_PATH

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    return data if isinstance(data, list) else []


def entry_to_puzzle(entry):
    return Puzzle(
        word=entry["word"].upper(),
        sentence=entry["sentence"],
        theme=entry["theme"],
        difficulty=entry["difficulty"],
        options=list(entry.get("options", [])),
        explanation=entry.get("explanation", ""),
        hint=entry.get("hint", ""),
    )


def load_bank(path=None):
    puzzles = []

    for entry in load_entries(path):
        try:
            puzzles.append(entry_to_puzzle(entry))
        except (KeyError, AttributeError, TypeError):
            continue  # skip malformed rows instead of crashing the game

    return puzzles


def pick_puzzle(
    theme,
    difficulty,
    used_words=None,
    length=None,
    path=None,
    rng=random,
):
    """
    Choose a puzzle, preferring: right theme + difficulty, not used yet.
    Falls back gracefully; returns None only if the bank has nothing usable.
    """
    used = {w.upper() for w in (used_words or [])}

    bank = load_bank(path)

    if length is not None:
        bank = [p for p in bank if len(p.word) == length]

    if not bank:
        return None

    # Unknown theme -> use the whole bank
    wanted = (theme or "").strip().casefold()

    same_theme = [p for p in bank if p.theme.casefold() == wanted] or bank

    same_difficulty = [p for p in same_theme if p.difficulty == difficulty]

    for candidates in (same_difficulty, same_theme):
        fresh = [p for p in candidates if p.word not in used]

        if fresh:
            return rng.choice(fresh)

    # Everything in this theme has been used: allow a repeat
    return rng.choice(same_difficulty or same_theme)


def themes(path=None):
    """Case-folded set of every theme that has at least one puzzle in the bank."""
    return {str(e.get("theme", "")).strip().casefold() for e in load_entries(path)}


def save_entries(entries, path=None):
    path = str(path or BANK_PATH)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    os.replace(tmp, path)  # atomic: a crash never leaves a half-written bank


def append_entry(entry, path=None):
    """
    Add one validated puzzle to the bank. Returns False (and writes nothing)
    if the word is already there. Safe to call from several sessions at once.
    """
    with _write_lock:
        entries = load_entries(path)

        if any(e.get("word", "").upper() == entry["word"].upper() for e in entries):
            return False

        entries.append(entry)
        save_entries(entries, path)

        return True
