"""
Builds data/lexicon.tsv and data/definitions.tsv (already shipped - you only
need to run this to rebuild them).

    pip install -r requirements-build.txt
    python scripts/build_lexicon.py

A word is kept only if ALL of these hold:
  - lowercase letters a-z only, 3-12 letters
  - it is a real dictionary word: WordNet knows it (or it is a regular
    inflection of one, e.g. plurals and verb forms)
  - WordNet has it as a common word, not only as a proper noun/name
  - it is not too rare (word-frequency filter) - no obscure or random tokens
"""

import sys
from pathlib import Path

import nltk
from wordfreq import top_n_list, zipf_frequency

nltk.download("wordnet", quiet=True)

from nltk.corpus import wordnet as wn  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"

MIN_LEN, MAX_LEN = 3, 12
MIN_ZIPF = 1.5           # keep anything at least this common
MIN_ZIPF_SHORT = 3.0     # 3-letter words: stricter (avoids abbreviations)
DEFINITION_MIN_ZIPF = 2.3

POS_ORDER = ["n", "v", "a", "r"]


def has_common_lemma(base):
    """True if WordNet lists `base` as an ordinary lowercase word (not a name)."""
    return any(
        lemma.name() == base
        for synset in wn.synsets(base)
        for lemma in synset.lemmas()
    )


def main():
    candidates = set(top_n_list("en", 300000))

    candidates.update(n for n in wn.all_lemma_names())

    rows = {}

    for word in candidates:
        if not (word.isascii() and word.isalpha() and word.islower()):
            continue

        if not MIN_LEN <= len(word) <= MAX_LEN:
            continue

        zipf = zipf_frequency(word, "en")

        if zipf < (MIN_ZIPF_SHORT if len(word) == 3 else MIN_ZIPF):
            continue

        base = wn.morphy(word) or (word if wn.synsets(word) else None)

        if not base or not base.isalpha() or not has_common_lemma(base):
            continue

        rows[word] = (round(zipf, 2), base)

    with open(DATA / "lexicon.tsv", "w", encoding="utf-8") as f:
        f.write("# word\tzipf\tlemma\n")

        for word in sorted(rows):
            zipf, base = rows[word]
            f.write(f"{word}\t{zipf}\t{base}\n")

    definitions = 0

    with open(DATA / "definitions.tsv", "w", encoding="utf-8") as f:
        f.write("# word\tpos\tdefinition\n")

        for word in sorted(rows):
            zipf, base = rows[word]

            if base != word or zipf < DEFINITION_MIN_ZIPF:
                continue

            synsets = wn.synsets(word)

            if not synsets:
                continue

            synset = sorted(
                synsets,
                key=lambda s: POS_ORDER.index(s.pos()) if s.pos() in POS_ORDER else 9,
            )[0]

            definition = synset.definition().replace("\t", " ").replace("\n", " ")

            f.write(f"{word}\t{synset.pos()}\t{definition}\n")
            definitions += 1

    print(f"lexicon.tsv: {len(rows)} words | definitions.tsv: {definitions} entries")


if __name__ == "__main__":
    sys.exit(main())
