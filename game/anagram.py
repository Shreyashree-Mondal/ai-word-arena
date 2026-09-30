"""
Anagram puzzle: unscramble the letters into a real word.

Correctness comes from the dictionary, not from string matching alone:
if the letters spell several real words (LISTEN / SILENT / TINSEL ...),
any of them is accepted.
"""

import random

from nlp import lexicon


def scramble(word, rng=random):
    """
    Shuffle the letters so the result is NOT the answer and NOT itself a
    real word (otherwise the puzzle would be confusing).
    """
    word = word.upper().strip()

    letters = list(word)

    for _ in range(200):
        rng.shuffle(letters)

        candidate = "".join(letters)

        if candidate != word and not lexicon.is_word(candidate):
            return candidate

    # Extremely rare (e.g. every permutation is a word): rotate instead
    rotated = word[1:] + word[:1]

    return rotated


def valid_answers(word):
    """All real words that use exactly these letters (the intended word included)."""
    word = word.upper().strip()

    return sorted(set(lexicon.anagrams_of(word)) | {word})


def check_anagram(puzzle, scrambled, guess):
    """
    Returns a dict:
      counts    - False for malformed input (wrong length / non-letters),
                  which does not cost an attempt
      correct   - True if the guess is a real word made from the letters
      message   - text to show the player
      answer    - the intended answer
    """
    answer = puzzle.word.upper().strip()
    scrambled = scrambled.upper().strip()
    guess = (guess or "").upper().strip()

    result = {"counts": False, "correct": False, "message": "", "answer": answer}

    if not guess.isalpha() or len(guess) != len(scrambled):
        result["message"] = f"Enter exactly {len(scrambled)} letters."
        return result

    result["counts"] = True

    if sorted(guess) != sorted(scrambled):
        result["message"] = "That doesn't use exactly the letters shown."
        return result

    if guess not in valid_answers(answer):
        result["message"] = "Right letters, but that is not a real word."
        return result

    result["correct"] = True

    if guess == answer:
        result["message"] = "Correct! 🎉"
    else:
        result["message"] = (
            f"Correct! 🎉 {guess} is a valid word too "
            f"(the intended answer was {answer})."
        )

    return result
