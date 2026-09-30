"""
Builds/extends data/puzzles.json using an LLM, with strict validation.

Every generated puzzle must pass the same checks the game relies on
(real 6-letter dictionary word, hint/sentence don't leak the answer,
exactly 3 sensible distractors), otherwise it is rejected.
"""

import json
import re
from collections import Counter

import config
from ai import llm_client
from ai.puzzle_bank import load_entries, save_entries
from ai.validator import validate_wordle
from nlp import lexicon

DIFFICULTY_RULES = {
    "Easy": "very common word that beginners know",
    "Medium": "moderately technical, commonly used in the field",
    "Hard": "advanced technical term that is less commonly known",
}


def build_prompt(theme, difficulty, count, avoid_words):
    avoid = ", ".join(sorted(avoid_words)) or "none"

    return f"""You are creating puzzles for an educational word game.

Theme: {theme}
Difficulty: {difficulty} ({DIFFICULTY_RULES.get(difficulty, '')})

Create {count} different puzzles. Return ONLY a JSON array, no other text.
Each item must have exactly these keys:
  "word":        the answer, EXACTLY {config.WORD_LENGTH} letters A-Z, a real English word related to the theme
  "hint":        one sentence clue that does NOT contain the word
  "sentence":    one sentence with the answer replaced by _____ (five underscores)
  "distractors": array of exactly 3 other real words that are wrong answers but plausible
  "explanation": one sentence explaining why the answer is correct

Rules:
- No abbreviations, acronyms, company names, people's names, hyphens or spaces.
- Do NOT use any of these words: {avoid}
- The word must not appear in "hint" or "sentence".
- Distractors must clearly NOT fit the sentence.

Example item:
{{"word": "SERVER", "hint": "A computer that delivers web pages to other computers.", "sentence": "A website's pages are sent to your browser by a _____.", "distractors": ["PRINTER", "MONITOR", "SCANNER"], "explanation": "A server provides data or services to client devices."}}"""


def parse_entries(text):
    """Extract a list of dicts from model output (tolerates code fences / chatter)."""
    if not text:
        return []

    text = re.sub(r"```(?:json)?", "", text)

    start, end = text.find("["), text.rfind("]")

    candidates = []

    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    obj_start, obj_end = text.find("{"), text.rfind("}")

    if obj_start != -1 and obj_end > obj_start:
        candidates.append(text[obj_start:obj_end + 1])

    for chunk in candidates:
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict):
            data = [data]

        if isinstance(data, list):
            found = [d for d in data if isinstance(d, dict)]

            # An inner list (e.g. "distractors") parses fine but holds no
            # puzzle dicts - keep looking instead of returning early.
            if found:
                return found

    return []


def validate_entry(raw, theme, difficulty, taken_words):
    """Returns (clean_entry | None, reason)."""
    try:
        word = str(raw["word"]).strip().upper()
        hint = str(raw["hint"]).strip()
        sentence = str(raw["sentence"]).strip()
        distractors = raw["distractors"]
        explanation = str(raw["explanation"]).strip()
    except (KeyError, TypeError):
        return None, "missing fields"

    ok, message = validate_wordle(word)

    if not ok:
        return None, message

    # Duplicates include inflections: SOCKET and SOCKETS are the same word
    if lexicon.is_duplicate(word, taken_words):
        return None, "duplicate word"

    if not hint or len(hint) > 200:
        return None, "bad hint"

    if lexicon.leaks_answer(hint, word):
        return None, "hint contains the answer"

    sentence = re.sub(r"_{3,}", "_____", sentence)

    if "_____" not in sentence:
        return None, "sentence has no blank"

    if lexicon.leaks_answer(sentence, word):
        return None, "sentence contains the answer"

    if not isinstance(distractors, list) or len(distractors) != 3:
        return None, "need exactly 3 distractors"

    clean_distractors = []

    for d in distractors:
        d = str(d).strip().upper()

        if not d.isalpha() or not lexicon.is_word(d):
            return None, "distractor is not a real word"

        if lexicon.is_duplicate(d, [word] + clean_distractors):
            return None, "bad distractors"

        clean_distractors.append(d)

    if not explanation or len(explanation) > 300:
        return None, "bad explanation"

    return (
        {
            "theme": theme,
            "difficulty": difficulty,
            "word": word,
            "hint": hint,
            "sentence": sentence,
            "options": [word] + clean_distractors,
            "explanation": explanation,
        },
        "ok",
    )


def build_bank(
    themes,
    difficulties,
    target,
    path=None,
    llm=None,
    batch_size=5,
    max_calls_per_combo=6,
    log=print,
):
    """
    Top up the bank so every theme+difficulty has at least `target` puzzles.
    `llm` is any function(prompt, max_tokens=..., temperature=...) -> str | None.
    """
    llm = llm or llm_client.generate_text

    entries = load_entries(path)

    stats = {"added": 0, "calls": 0, "rejected": Counter(), "aborted": False}

    for theme in themes:
        for difficulty in difficulties:

            def have():
                return sum(
                    1
                    for e in entries
                    if e.get("theme") == theme and e.get("difficulty") == difficulty
                )

            calls = 0

            while have() < target and calls < max_calls_per_combo:
                calls += 1
                stats["calls"] += 1

                need = min(batch_size, target - have())

                taken = {e["word"].upper() for e in entries}

                text = llm(
                    build_prompt(theme, difficulty, need, taken),
                    max_tokens=900,
                    temperature=0.8,
                )

                if text is None:
                    stats["aborted"] = True
                    log("AI unavailable (no response) - stopping. Progress is saved.")
                    save_entries(entries, path)
                    return stats

                for raw in parse_entries(text):
                    entry, reason = validate_entry(
                        raw, theme, difficulty, {e["word"].upper() for e in entries}
                    )

                    if entry is None:
                        stats["rejected"][reason] += 1
                        continue

                    entries.append(entry)
                    stats["added"] += 1

            save_entries(entries, path)

            log(f"{theme} / {difficulty}: {have()}/{target}")

    return stats
