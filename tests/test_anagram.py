import random

import pytest

from database.models import Puzzle
from game.anagram import check_anagram, scramble, valid_answers
from nlp import lexicon


def puzzle(word):
    return Puzzle(word=word, sentence="_____", theme="Space", difficulty="Easy")


@pytest.mark.parametrize("word", ["ROCKET", "PLANET", "LISTEN", "SERVER", "BINARY"])
def test_scramble_keeps_letters_and_is_never_a_real_word(word):
    for seed in range(25):
        s = scramble(word, random.Random(seed))

        assert sorted(s) == sorted(word)
        assert s != word
        assert not lexicon.is_word(s)


def test_scramble_is_reproducible_with_a_seed():
    assert scramble("ROCKET", random.Random(1)) == scramble("ROCKET", random.Random(1))


def test_correct_answer():
    r = check_anagram(puzzle("ROCKET"), "TEROCK", "rocket")

    assert r["counts"] and r["correct"] and "Correct" in r["message"]


def test_any_real_word_from_the_letters_is_accepted():
    p = puzzle("LISTEN")

    assert "SILENT" in valid_answers("LISTEN")

    r = check_anagram(p, "TNESIL", "SILENT")

    assert r["correct"] is True
    assert "valid word too" in r["message"] and "LISTEN" in r["message"]


def test_wrong_letters_and_non_words_cost_an_attempt_but_are_not_correct():
    p = puzzle("ROCKET")

    wrong_letters = check_anagram(p, "TEROCK", "ROCKER")
    assert wrong_letters["counts"] and not wrong_letters["correct"]
    assert "letters" in wrong_letters["message"]

    not_a_word = check_anagram(p, "TEROCK", "KOCRET")
    assert not_a_word["counts"] and not not_a_word["correct"]
    assert "not a real word" in not_a_word["message"]


def test_malformed_input_costs_no_attempt():
    p = puzzle("ROCKET")

    for bad in ["", "ROC", "ROCKET1", "ROCKETS"]:
        r = check_anagram(p, "TEROCK", bad)

        assert r["counts"] is False and r["correct"] is False
