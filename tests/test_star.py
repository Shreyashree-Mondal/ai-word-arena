import random

import pytest

from game import star
from nlp import lexicon


def make(seed_word="MOUNTAIN", center="N"):
    puzzle = star.build_star(seed_word, center)
    assert puzzle is not None
    return puzzle


@pytest.mark.parametrize("difficulty", ["Easy", "Medium", "Hard"])
def test_generated_puzzles_satisfy_every_rule(difficulty):
    params = star.DIFFICULTY_PARAMS[difficulty]

    for seed in range(8):
        p = star.generate_star(difficulty, random.Random(seed))

        assert p is not None

        letters = p.letters

        assert len(letters) == 7 and "S" not in letters
        assert params["min_words"] <= len(p.words) <= params["max_words"]
        assert p.pangrams and all(set(w) == set(letters) for w in p.pangrams)
        assert len(p.words) == len(set(p.words))                      # no duplicates
        assert p.max_points == sum(star.score_word(w, letters) for w in p.words)

        for w in p.words:
            assert len(w) >= 4 and p.center in w and set(w) <= letters
            assert lexicon.is_word(w), f"{w} is not a real word"


def test_generation_is_reproducible_with_a_seed():
    a = star.generate_star("Medium", random.Random(7))
    b = star.generate_star("Medium", random.Random(7))

    assert (a.seed, a.center, a.words) == (b.seed, b.center, b.words)


def test_used_seeds_and_their_inflections_are_avoided():
    first = star.generate_star("Medium", random.Random(3))

    for seed in range(15):
        again = star.generate_star("Medium", random.Random(seed), used_seeds=[first.seed])

        assert not lexicon.is_duplicate(again.seed, [first.seed])


def test_scoring_and_ranks():
    letters = set("MOUNTAI")

    assert star.score_word("MAIN", letters) == 1          # 4 letters = 1 point
    assert star.score_word("MOUNT", letters) == 5         # longer = its length
    assert star.score_word("MOUNTAIN", letters) == 8 + 7  # pangram bonus

    assert star.rank_for(0, 100) == "Beginner"
    assert star.rank_for(35, 100) == "Good"
    assert star.rank_for(100, 100) == "Genius"
    assert star.rank_for(0, 0) == "Beginner"


def test_check_star_word_gives_specific_feedback():
    p = make()
    ok_word = "MOUNT"

    assert ok_word in p.word_set

    assert star.check_star_word(p, [], "mount")["ok"] is True
    assert star.check_star_word(p, [], "mount")["points"] == 5
    assert star.check_star_word(p, [], "MOUNTAIN")["pangram"] is True

    assert "Too short" in star.check_star_word(p, [], "MAN")["message"]
    assert "center letter" in star.check_star_word(p, [], "MAIM")["message"]
    assert "not in the star" in star.check_star_word(p, [], "TOWN")["message"]
    assert "already found" in star.check_star_word(p, ["MOUNT"], "mount")["message"]
    assert "Not a valid English word" in star.check_star_word(p, [], "NNNNN")["message"]
    assert "Letters only" in star.check_star_word(p, [], "MO3NT")["message"]


def test_valid_english_word_outside_the_list_is_told_apart_from_a_non_word():
    p = make()

    rare = next(
        w for w in lexicon.all_words(4, lexicon.GUESS_MIN_ZIPF)
        if set(w) <= p.letters and p.center in w and w not in p.word_set
    )

    assert "not in this puzzle's word list" in star.check_star_word(p, [], rare)["message"]


def test_hints_narrow_things_down_without_revealing_a_word():
    p = make()

    first = star.star_hint(p, [], hint_number=0)

    # first hint points at a pangram: gives its length and first letter only
    target = next(w for w in p.pangrams if f"{len(w)} letters" in first and f"'{w[0]}'" in first)

    assert "pangram" in first and target not in first.upper()

    for i in range(20):
        hint = star.star_hint(p, [], hint_number=1, rng=random.Random(i)).upper()

        assert not any(w in hint for w in p.words), hint   # never contains a whole answer

    assert "found every word" in star.star_hint(p, list(p.words), hint_number=1)


def test_ai_seed_suggestions_are_verified_by_code():
    fake = "TELESCOPE, PLANETS, ROCKET, SPACING, JOURNAL, ZZZZZZZ, MOUNTAIN, journal"

    words = star.suggest_seed_words("Space", llm=lambda *a, **k: fake)

    assert words == ["JOURNAL", "MOUNTAIN"]   # S-words, short words, non-words, repeats all dropped
    assert star.suggest_seed_words("Space", llm=lambda *a, **k: None) == []


def test_ai_seed_is_used_when_it_works():
    llm = lambda *a, **k: "JOURNAL, MOUNTAIN"  # noqa: E731

    p = star.get_star_puzzle("Space", "Hard", rng=random.Random(1), llm=llm)

    assert p.source == "ai" and p.seed in {"JOURNAL", "MOUNTAIN"} and p.theme == "Space"


@pytest.mark.parametrize("reply", [None, "no idea", "ZZZZZZZ, QQQQQQQ", "TELESCOPE, PLANETS"])
def test_ai_failure_or_junk_falls_back_to_the_dictionary(reply):
    p = star.get_star_puzzle("Space", "Medium", rng=random.Random(1), llm=lambda *a, **k: reply)

    assert p is not None and p.source == "lexicon"


def test_ai_crash_falls_back_to_the_dictionary():
    def boom(*a, **k):
        raise RuntimeError("network down")

    p = star.get_star_puzzle("Space", "Medium", rng=random.Random(1), llm=boom)

    assert p.source == "lexicon"


def test_no_ai_means_dictionary_puzzle(monkeypatch):
    from ai import llm_client

    monkeypatch.setattr(llm_client, "is_available", lambda: False)

    assert star.get_star_puzzle("Space", "Easy", rng=random.Random(2)).source == "lexicon"
