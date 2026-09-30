"""
The agent's tools. They are plain deterministic code: the LLM decides WHICH
tool to call, but the answers come from the dictionary, never from the model.
"""

import re

from ai.bank_builder import validate_entry
from ai.validator import validate_wordle
from nlp import lexicon

TIERS = ["Easy", "Medium", "Hard"]

# Word-frequency bands. On the shipped bank these thresholds agree with the
# hand-labelled difficulty for 71% of puzzles, and are within one level for 95%.
EASY_MIN_ZIPF = 4.0
MEDIUM_MIN_ZIPF = 3.6

TOOL_DOCS = {
    "check_words": (
        '{"words": ["..", ".."]} - Check up to 10 candidate answers AT ONCE: which are '
        "usable (6-letter real base-form words, not used yet) and their difficulty."
    ),
    "check_word": '{"word": ".."} - Check one word in detail (tier, definition, already used?).',
    "check_duplicate": '{"word": ".."} - Was this word (or a plural) already used?',
    "check_leak": '{"text": "..", "word": ".."} - Does the text give the answer away?',
    "estimate_difficulty": '{"word": ".."} - Difficulty tier of a word vs the target.',
    "check_distractors": '{"word": "..", "distractors": [".."]} - Are the wrong options real words?',
    "lookup_definition": '{"word": ".."} - Dictionary definition.',
    "submit_puzzle": (
        '{"word", "hint", "sentence" (with _____), "distractors" (3 words), "explanation"} '
        "- FINAL answer. Code validates it and says what to fix."
    ),
}

MAX_BATCH = 10


def tier_for(word):
    z = lexicon.zipf(word)

    if z >= EASY_MIN_ZIPF:
        return "Easy"

    if z >= MEDIUM_MIN_ZIPF:
        return "Medium"

    return "Hard"


def tier_distance(word, target):
    if target not in TIERS:
        return 0

    return abs(TIERS.index(tier_for(word)) - TIERS.index(target))


def _word_arg(args):
    return str(args.get("word", "")).strip().upper()


class ToolBox:
    def __init__(self, theme, difficulty, used_words=(), bank_words=()):
        self.theme = theme
        self.difficulty = difficulty
        self.used_words = [w.upper() for w in used_words]
        self.bank_words = [w.upper() for w in bank_words]
        self.avoid = None   # optional: words to steer the model away from in the prompt

    @property
    def taken(self):
        return self.used_words + self.bank_words

    def names(self):
        return list(TOOL_DOCS)

    # ---------------- tools ----------------

    def check_word(self, args):
        word = _word_arg(args)

        valid, reason = validate_wordle(word)

        used = lexicon.is_duplicate(word, self.taken) if word else False

        result = {
            "word": word,
            "valid_answer": valid,
            "reason": reason,
            "already_used": used,
            "usable": bool(valid and not used),
        }

        if valid and used:
            result["advice"] = "Already used - choose a different word."

        if valid:
            result.update(
                zipf=round(lexicon.zipf(word), 2),
                tier=tier_for(word),
                target=self.difficulty,
                part_of_speech=lexicon.part_of_speech(word),
                definition=(lexicon.definition(word) or "")[:120],
            )

        return result

    def check_words(self, args):
        """Check many candidates in ONE call - far cheaper than one step per word."""
        words = args.get("words")

        if isinstance(words, str):
            words = re.split(r"[,\s]+", words)

        if not isinstance(words, list):
            return {"error": f"words must be a list of up to {MAX_BATCH} words"}

        usable, rejected, seen = [], [], set()

        for raw in words[:MAX_BATCH]:
            word = str(raw).strip().upper()

            if not word or word in seen:
                continue

            seen.add(word)

            info = self.check_word({"word": word})

            if info["usable"]:
                usable.append(
                    {
                        "word": word,
                        "tier": info["tier"],
                        "fits_difficulty": tier_distance(word, self.difficulty) <= 1,
                    }
                )
            else:
                why = "already used" if info["valid_answer"] else info["reason"]
                rejected.append({"word": word, "why": why})

        return {"usable": usable, "rejected": rejected}

    def check_duplicate(self, args):
        word = _word_arg(args)

        clashes = [t for t in self.taken if lexicon.same_word(word, t)]

        return {"word": word, "duplicate": bool(clashes), "conflicts_with": clashes[:3]}

    def check_leak(self, args):
        word = _word_arg(args)
        text = str(args.get("text", ""))

        return {"leaks": lexicon.leaks_answer(text, word)}

    def estimate_difficulty(self, args):
        word = _word_arg(args)

        if not lexicon.is_word(word):
            return {"word": word, "error": "not a real word"}

        distance = tier_distance(word, self.difficulty)

        return {
            "word": word,
            "zipf": round(lexicon.zipf(word), 2),
            "tier": tier_for(word),
            "target": self.difficulty,
            "acceptable": distance <= 1,
        }

    def check_distractors(self, args):
        word = _word_arg(args)
        options = args.get("distractors")

        if not isinstance(options, list):
            return {"error": "distractors must be a list of 3 words"}

        report, problems, seen = [], [], []

        for d in options:
            d = str(d).strip().upper()

            real = lexicon.is_word(d)
            same = bool(word) and lexicon.same_word(d, word)
            repeat = lexicon.is_duplicate(d, seen)

            report.append({"word": d, "real_word": real})

            if not real:
                problems.append(f"{d} is not a real word")
            if same:
                problems.append(f"{d} is a form of the answer")
            if repeat:
                problems.append(f"{d} is repeated")

            seen.append(d)

        if len(options) != 3:
            problems.append("need exactly 3 distractors")

        return {"options": report, "problems": problems, "ok": not problems}

    def lookup_definition(self, args):
        word = _word_arg(args)

        definition = lexicon.definition(word)

        if not definition:
            return {"word": word, "found": False}

        return {
            "word": word,
            "found": True,
            "part_of_speech": lexicon.part_of_speech(word),
            "definition": definition[:160],
        }

    # ---------------- dispatch ----------------

    def call(self, name, args):
        if name not in TOOL_DOCS or name == "submit_puzzle":
            return {"error": f"Unknown tool '{name}'. Available: {', '.join(self.names())}"}

        try:
            return getattr(self, name)(args if isinstance(args, dict) else {})
        except Exception as exc:  # a broken tool call must never crash the agent
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ---------------- final gate ----------------

    def validate_submission(self, args):
        """
        The hard gate. Whatever the agent believes, a puzzle is accepted only
        if it passes the same strict checks as every other puzzle.
        Returns (public_verdict, clean_entry_or_None).
        """
        args = args if isinstance(args, dict) else {}

        raw = {
            key: args.get(key)
            for key in ("word", "hint", "sentence", "distractors", "explanation")
        }

        entry, reason = validate_entry(raw, self.theme, self.difficulty, self.taken)

        if entry is None:
            return {"accepted": False, "problems": [reason]}, None

        if tier_distance(entry["word"], self.difficulty) >= 2:
            return (
                {
                    "accepted": False,
                    "problems": [
                        f"{entry['word']} is a {tier_for(entry['word'])}-level word "
                        f"but the target is {self.difficulty}; pick a closer word"
                    ],
                },
                None,
            )

        return {"accepted": True, "problems": []}, entry
