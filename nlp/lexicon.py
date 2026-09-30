"""
The single source of truth for "is this a real word?".

Backed by data/lexicon.tsv (built from WordNet + word frequencies, see
scripts/build_lexicon.py). Nothing here calls an AI - it is deterministic,
so a puzzle word is never something a model merely made up.
"""

import functools
import re
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

LEXICON_PATH = DATA / "lexicon.tsv"
DEFINITIONS_PATH = DATA / "definitions.tsv"

# A player's GUESS may be any real word (including plurals / verb forms)
GUESS_MIN_ZIPF = 1.5

# A puzzle ANSWER must be a base-form word that is not obscure
ANSWER_MIN_ZIPF = 2.3


@functools.lru_cache(maxsize=1)
def _load():
    """word(UPPER) -> (zipf, lemma UPPER)"""
    words = {}

    try:
        with open(LEXICON_PATH, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue

                word, zipf, lemma = line.rstrip("\n").split("\t")

                words[word.upper()] = (float(zipf), lemma.upper())
    except OSError:
        pass

    return words


@functools.lru_cache(maxsize=1)
def _load_definitions():
    defs = {}

    try:
        with open(DEFINITIONS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue

                word, pos, definition = line.rstrip("\n").split("\t", 2)

                defs[word.upper()] = (pos, definition)
    except OSError:
        pass

    return defs


def is_loaded():
    return bool(_load())


# ---------------------------------------
# Lookups
# ---------------------------------------

def _norm(word):
    return (word or "").strip().upper()


def is_word(word, min_zipf=GUESS_MIN_ZIPF):
    entry = _load().get(_norm(word))

    return entry is not None and entry[0] >= min_zipf


def zipf(word):
    entry = _load().get(_norm(word))

    return entry[0] if entry else 0.0


def lemma(word):
    """Base form (SOCKETS -> SOCKET). Unknown words are returned unchanged."""
    word = _norm(word)
    entry = _load().get(word)

    return entry[1] if entry else word


def same_word(a, b):
    """True if two words are the same word or inflections of one another."""
    return lemma(a) == lemma(b)


def is_duplicate(word, taken):
    """True if `word` (or an inflection of it) is already in `taken`."""
    target = lemma(word)

    return any(lemma(t) == target for t in taken)


def is_answer_word(word):
    """Strict check for puzzle answers. Returns (ok, reason)."""
    word = _norm(word)

    if not word.isalpha():
        return False, "Word must contain only alphabetic characters."

    entry = _load().get(word)

    if entry is None:
        return False, "Not a valid English word."

    if entry[0] < ANSWER_MIN_ZIPF:
        return False, "Too obscure to be a fair puzzle word."

    if entry[1] != word:
        return False, "Use the base form of the word (no plurals or verb forms)."

    return True, "Valid word."


def definition(word):
    """Dictionary definition of a word, or None."""
    entry = _load_definitions().get(_norm(word))

    return entry[1] if entry else None


def part_of_speech(word):
    entry = _load_definitions().get(_norm(word))

    return entry[0] if entry else None


# ---------------------------------------
# Word lists
# ---------------------------------------

def all_words(min_len=1, min_zipf=GUESS_MIN_ZIPF):
    """Every word in the lexicon at least this long and this common (UPPERCASE)."""
    return sorted(
        w for w, (z, _l) in _load().items() if len(w) >= min_len and z >= min_zipf
    )


def words_of_length(length, min_zipf=GUESS_MIN_ZIPF):
    return sorted(
        w for w, (z, _l) in _load().items() if len(w) == length and z >= min_zipf
    )


@functools.lru_cache(maxsize=1)
def _anagram_index():
    index = defaultdict(list)

    for word in _load():
        index["".join(sorted(word))].append(word)

    return index


def anagrams_of(letters, min_zipf=GUESS_MIN_ZIPF):
    """Every real word that uses exactly these letters (any order)."""
    key = "".join(sorted(_norm(letters)))

    return sorted(
        w for w in _anagram_index().get(key, []) if _load()[w][0] >= min_zipf
    )


# ---------------------------------------
# Answer-leak detection
# ---------------------------------------

_TOKEN = re.compile(r"[A-Za-z]+")


def leaks_answer(text, answer):
    """
    True if `text` gives the answer away: the word itself, an inflection of
    it (rocket / rockets), or a compound containing it (rocketship).
    """
    answer = _norm(answer)
    answer_lemma = lemma(answer)

    if not answer:
        return False

    if answer.lower() in (text or "").lower():
        return True

    for token in _TOKEN.findall(text or ""):
        token = token.upper()

        if token == answer or lemma(token) == answer_lemma:
            return True

    return False
