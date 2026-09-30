from database.models import Puzzle
from game.context_quiz import check_answer, check_wordle_answer


def make(word):
    return Puzzle(word=word, sentence="_____", theme="Space", difficulty="Easy")


def test_mcq_correct_and_wrong():
    p = make("ROCKET")

    assert check_answer(p, "rocket").correct is True
    assert check_answer(p, "TRAIN").correct is False


def test_wordle_all_green_is_correct():
    r = check_wordle_answer(make("ROCKET"), "ROCKET")

    assert r["valid"] and r["correct"]
    assert r["feedback"] == ["green"] * 6


def test_wordle_duplicate_letters_only_marked_once():
    # answer has ONE 'E'; guess has two -> only one may be coloured
    r = check_wordle_answer(make("PLANET"), "SCREEN")

    assert r["valid"]
    assert r["feedback"].count("yellow") + r["feedback"].count("green") == 2  # E and N


def test_wordle_rejects_non_words():
    r = check_wordle_answer(make("ROCKET"), "QQQQQQ")

    assert r["valid"] is False
    assert r["correct"] is False
