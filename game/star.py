"""
Star puzzle (Spelling-Bee style).

Seven letters, one of them in the centre. Find words of 4+ letters that
use ONLY those letters and ALWAYS include the centre letter. At least one
word (the pangram) uses all seven letters.

Every solution word is computed from the dictionary, so it is a real word
by construction - the AI may only suggest a themed pangram, and even that
is verified by code before it is used.
"""

import functools
import random
import re
from dataclasses import dataclass, field
from typing import List

import config
from ai import llm_client
from database.models import Puzzle
from nlp import lexicon

MIN_WORD_LEN = 4
LETTER_COUNT = 7
SOLUTION_MIN_ZIPF = 2.0     # words that count towards the puzzle's total
PANGRAM_BONUS = 7
WIN_PERCENT = 35            # "Good" rank or better counts as a win

DIFFICULTY_PARAMS = {
    "Easy":   {"min_words": 25, "max_words": 90, "seed_min_zipf": 3.5},
    "Medium": {"min_words": 18, "max_words": 60, "seed_min_zipf": 3.0},
    "Hard":   {"min_words": 12, "max_words": 40, "seed_min_zipf": 2.5},
}

RANKS = [
    (0, "Beginner"),
    (10, "Good Start"),
    (20, "Moving Up"),
    (35, "Good"),
    (50, "Solid"),
    (65, "Nice"),
    (80, "Great"),
    (90, "Amazing"),
    (100, "Genius"),
]


@dataclass
class StarPuzzle:
    center: str
    outer: List[str]
    seed: str
    words: List[str]
    pangrams: List[str]
    max_points: int
    theme: str = ""
    source: str = "lexicon"          # "ai" (themed seed) or "lexicon"
    word_set: frozenset = field(default_factory=frozenset)

    @property
    def letters(self):
        return frozenset([self.center] + list(self.outer))


# ---------------------------------------
# Scoring
# ---------------------------------------

def score_word(word, letters):
    points = 1 if len(word) == MIN_WORD_LEN else len(word)

    if set(word) == set(letters):
        points += PANGRAM_BONUS

    return points


def rank_for(points, max_points):
    percent = 100 * points / max_points if max_points else 0

    name = RANKS[0][1]

    for threshold, label in RANKS:
        if percent >= threshold:
            name = label

    return name


# ---------------------------------------
# Building a puzzle from a seed word
# ---------------------------------------

@functools.lru_cache(maxsize=1)
def _pool():
    """(word, set_of_letters) for every usable solution word."""
    return [
        (w, frozenset(w))
        for w in lexicon.all_words(MIN_WORD_LEN, SOLUTION_MIN_ZIPF)
        if len(set(w)) <= LETTER_COUNT
    ]


def is_valid_seed(word, min_zipf=0.0, allow_s=False):
    word = (word or "").strip().upper()

    if not (word.isalpha() and 7 <= len(word) <= 9):
        return False

    if len(set(word)) != LETTER_COUNT:
        return False

    if not allow_s and "S" in word:
        return False  # trailing-S plurals make puzzles too easy

    if not lexicon.is_answer_word(word)[0]:
        return False

    return lexicon.zipf(word) >= min_zipf


def build_star(seed, center, theme="", source="lexicon"):
    seed = seed.upper()
    center = center.upper()

    letters = frozenset(seed)

    if len(letters) != LETTER_COUNT or center not in letters:
        return None

    words = sorted(w for w, s in _pool() if center in s and s <= letters)

    pangrams = [w for w in words if set(w) == letters]

    if not pangrams:
        return None

    outer = sorted(letters - {center})

    return StarPuzzle(
        center=center,
        outer=outer,
        seed=seed,
        words=words,
        pangrams=pangrams,
        max_points=sum(score_word(w, letters) for w in words),
        theme=theme,
        source=source,
        word_set=frozenset(words),
    )


def _seed_candidates(min_zipf, allow_s):
    return [
        w for w in lexicon.all_words(7, min_zipf)
        if len(w) <= 9 and is_valid_seed(w, min_zipf, allow_s)
    ]


def generate_star(
    difficulty="Medium",
    rng=random,
    seeds=None,
    used_seeds=(),
    theme="",
    source="lexicon",
    max_tries=60,
):
    """Returns a StarPuzzle within the difficulty's word-count range, or None."""
    params = DIFFICULTY_PARAMS.get(difficulty, DIFFICULTY_PARAMS["Medium"])

    if seeds is not None:
        candidates = [s.upper() for s in seeds if is_valid_seed(s)]
    else:
        candidates = _seed_candidates(params["seed_min_zipf"], allow_s=False)

    candidates = [s for s in candidates if not lexicon.is_duplicate(s, used_seeds)]

    rng.shuffle(candidates)

    for seed in candidates[:max_tries]:
        centers = sorted(set(seed))
        rng.shuffle(centers)

        for center in centers:
            star = build_star(seed, center, theme, source)

            if star and params["min_words"] <= len(star.words) <= params["max_words"]:
                rng.shuffle(star.outer)
                return star

    return None


# ---------------------------------------
# Optional AI: suggest themed pangram seeds (verified by code)
# ---------------------------------------

def suggest_seed_words(theme, llm=None, count=12):
    llm = llm or llm_client.generate_text

    text = llm(
        f"List {count} common English words related to '{theme}' that each have "
        "exactly 7 different letters and no letter S. "
        "Return only the words, separated by commas.",
        max_tokens=150,
        temperature=0.8,
        timeout=config.LIVE_AI_TIMEOUT,
    )

    if not text:
        return []

    seen, words = set(), []

    for token in re.findall(r"[A-Za-z]+", text.upper()):
        if token not in seen and is_valid_seed(token):
            seen.add(token)
            words.append(token)

    return words


def get_star_puzzle(theme, difficulty, used_seeds=(), rng=random, llm=None):
    """
    Themed AI seed first (if AI is connected), otherwise - or if that fails -
    a seed straight from the dictionary. Never returns an unverified puzzle.
    """
    if llm is not None or llm_client.is_available():
        try:
            seeds = suggest_seed_words(theme, llm)
        except Exception as exc:
            print(f"[star] AI seed suggestion failed: {exc}")
            seeds = []

        if seeds:
            star = generate_star(
                difficulty, rng, seeds=seeds, used_seeds=used_seeds,
                theme=theme, source="ai",
            )

            if star:
                return star

    return generate_star(
        difficulty, rng, used_seeds=used_seeds, theme=theme, source="lexicon"
    )


def as_puzzle(star, difficulty):
    """Lightweight Puzzle wrapper so the shared round logic can track this round."""
    return Puzzle(
        word=star.seed,
        sentence="",
        theme=star.theme,
        difficulty=difficulty,
        source=star.source,
    )


# ---------------------------------------
# Checking words and hints
# ---------------------------------------

def check_star_word(star, found, raw):
    """Returns {ok, message, points, pangram}."""
    word = (raw or "").strip().upper()

    result = {"ok": False, "message": "", "points": 0, "pangram": False}

    if not word.isalpha():
        result["message"] = "Letters only, please."
    elif len(word) < MIN_WORD_LEN:
        result["message"] = f"Too short - words need {MIN_WORD_LEN}+ letters."
    elif star.center not in word:
        result["message"] = f"Every word must use the center letter {star.center}."
    elif any(ch not in star.letters for ch in word):
        result["message"] = "That uses a letter that is not in the star."
    elif word in {f.upper() for f in found}:
        result["message"] = f"You already found {word}."
    elif word not in star.word_set:
        if lexicon.is_word(word):
            result["message"] = "A real word, but not in this puzzle's word list."
        else:
            result["message"] = "Not a valid English word."
    else:
        points = score_word(word, star.letters)

        result.update(
            ok=True,
            points=points,
            pangram=set(word) == set(star.letters),
            message=f"+{points} points",
        )

    return result


def star_hint(star, found, hint_number=0, rng=random):
    """A hint that narrows things down without giving a whole word away."""
    found_upper = {f.upper() for f in found}

    missing_pangrams = [p for p in star.pangrams if p not in found_upper]

    if hint_number == 0 and missing_pangrams:
        p = missing_pangrams[0]

        return (
            f"There is a pangram (it uses all 7 letters) with {len(p)} letters, "
            f"starting with '{p[0]}'."
        )

    unfound = [w for w in star.words if w not in found_upper]

    if not unfound:
        return "You have found every word!"

    w = rng.choice(unfound)

    return f"Try a {len(w)}-letter word starting with '{w[:2]}'."
