"""Categories, icons and safe handling of user-typed custom topics."""

import re

import config
from ai import puzzle_bank

ICONS = {
    "Technology": "💻",
    "Science": "🔬",
    "Healthcare": "🩺",
    "Finance": "💰",
    "Environment": "🌿",
    "Space": "🚀",
    "Engineering": "🛠️",
    "Education": "🎓",
    "History": "🏛️",
    "Arts & Culture": "🎨",
    "Cricket": "🏏",
    "Entertainment": "🎬",
    "Sports": "⚽",
    "Food & Cooking": "🍳",
    "Music": "🎵",
    "Geography": "🌍",
    "Animals": "🐾",
    "Mathematics": "➗",
}

CUSTOM_LABEL = "✍️ Custom topic..."

MAX_TOPIC_LEN = 40

# Letters, digits, spaces and a few harmless symbols only. Quotes, newlines,
# braces and other prompt-breaking characters are removed because the topic is
# inserted into AI prompts.
_DISALLOWED = re.compile(r"[^A-Za-z0-9 &'\-]")


def label(theme):
    return f"{ICONS.get(theme, '🎯')} {theme}"


def clean_topic(text):
    """
    Returns a safe topic string ("" if unusable). Typing an existing category
    in any letter case returns that category's proper name.
    """
    text = _DISALLOWED.sub("", text or "")
    text = re.sub(r"\s+", " ", text).strip("-' ").strip()[:MAX_TOPIC_LEN].strip()

    if sum(ch.isalpha() for ch in text) < 2:
        return ""

    for sector in config.SECTORS:
        if sector.casefold() == text.casefold():
            return sector

    return text[0].upper() + text[1:]


def has_offline_puzzles(theme):
    """True if this topic can be played without the AI (a category, or already in the bank)."""
    wanted = (theme or "").strip().casefold()

    if not wanted:
        return False

    if any(s.casefold() == wanted for s in config.SECTORS):
        return True

    return wanted in puzzle_bank.themes()
