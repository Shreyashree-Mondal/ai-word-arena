"""Word validation for puzzles and guesses - powered by the strict lexicon."""

from config import WORD_LENGTH
from nlp import lexicon


def _basic_checks(word):
    word = (word or "").strip().upper()

    if len(word) != WORD_LENGTH:
        return None, (
            False,
            f"Word must contain exactly {WORD_LENGTH} letters.",
        )

    if not word.isalpha():
        return None, (
            False,
            "Word must contain only alphabetic characters.",
        )

    return word, None


# --------------------------------
# Validate a puzzle ANSWER (strict: base form, not obscure)
# --------------------------------

def validate_wordle(word):
    word, error = _basic_checks(word)

    if error:
        return error

    return lexicon.is_answer_word(word)


# --------------------------------
# Validate a player's GUESS (any real word, plurals and verb forms allowed)
# --------------------------------

def validate_guess(word):
    word, error = _basic_checks(word)

    if error:
        return error

    if not lexicon.is_word(word):
        return False, "Not a valid English word."

    return True, "Valid word."

