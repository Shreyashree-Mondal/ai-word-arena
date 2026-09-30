import pytest

from ai.puzzle_bank import load_entries
from ai.validator import validate_guess, validate_wordle
from nlp import lexicon


def test_lexicon_is_loaded():
    assert lexicon.is_loaded()


@pytest.mark.parametrize("junk", ["TEH", "JAMES", "OBAMA", "QQQQQQ", "ZZZZZZ", "XYZABC"])
def test_typos_names_brands_and_random_strings_are_rejected(junk):
    assert not lexicon.is_word(junk)


@pytest.mark.parametrize("word", ["ROCKET", "QUASAR", "PULSAR", "PYTHON", "ENZYME", "ZENITH"])
def test_real_and_modern_words_are_accepted(word):
    assert lexicon.is_word(word)


def test_guesses_may_be_inflected_but_answers_must_be_base_forms():
    assert validate_guess("PLANTS")[0] is True     # a fair guess
    assert validate_wordle("PLANTS")[0] is False   # not a valid puzzle answer
    assert validate_wordle("PLANET")[0] is True


def test_length_and_characters_are_enforced():
    assert validate_wordle("CAT")[0] is False
    assert validate_wordle("HELLO1")[0] is False
    assert validate_guess("")[0] is False


def test_duplicates_include_inflections():
    assert lexicon.is_duplicate("SOCKETS", ["SOCKET"])
    assert lexicon.is_duplicate("rocket", ["ROCKETS"])
    assert not lexicon.is_duplicate("KERNEL", ["SOCKET"])
    assert lexicon.same_word("PLANETS", "planet")


def test_leak_detection_understands_word_forms():
    assert lexicon.leaks_answer("A rocket flies to space.", "ROCKET")
    assert lexicon.leaks_answer("Many rockets were launched.", "ROCKET")   # inflection
    assert lexicon.leaks_answer("Board the rocketship.", "ROCKET")         # compound
    assert not lexicon.leaks_answer("A vehicle that reaches orbit.", "ROCKET")


def test_anagram_lookup_finds_all_real_words():
    assert {"LISTEN", "SILENT", "TINSEL", "ENLIST", "INLETS"} <= set(lexicon.anagrams_of("LISTEN"))
    assert lexicon.anagrams_of("QQQQQQ") == []


def test_definitions_exist_for_common_words():
    assert lexicon.definition("planet")
    assert lexicon.definition("qqqqqq") is None


def test_whole_bank_is_dictionary_correct_with_no_duplicates():
    entries = load_entries()
    seen = set()

    for e in entries:
        word = e["word"]

        assert lexicon.is_answer_word(word)[0], f"{word} is not a strict dictionary word"

        assert lexicon.lemma(word) not in seen, f"{word} duplicates another puzzle"
        seen.add(lexicon.lemma(word))

        assert not lexicon.leaks_answer(e["hint"], word), word
        assert not lexicon.leaks_answer(e["sentence"], word), word

        for option in e["options"]:
            assert lexicon.is_word(option), f"option {option} (for {word}) is not a real word"

        assert len({lexicon.lemma(o) for o in e["options"]}) == 4, f"{word}: options overlap"
