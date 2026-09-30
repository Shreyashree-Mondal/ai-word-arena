from collections import Counter

from database.models import Puzzle, GameResult
from ai.validator import validate_guess


QUIZ_POINTS = 10


# --------------------------------
# MCQ Answer Checking
# --------------------------------
def check_answer(
    puzzle: Puzzle,
    selected_answer: str
) -> GameResult:
    """
    Check player's MCQ answer.
    """

    selected_answer = selected_answer.upper().strip()
    correct_answer = puzzle.word.upper().strip()

    if selected_answer == correct_answer:

        return GameResult(
            correct=True,
            points=QUIZ_POINTS,
            message="Correct! 🎉"
        )

    return GameResult(
        correct=False,
        points=0,
        message=f"Wrong answer. The correct word was {puzzle.word}."
    )


# --------------------------------
# Wordle Answer Checking
# --------------------------------
def check_wordle_answer(puzzle, guess):

    answer = puzzle.word.upper().strip()
    guess = guess.upper().strip()

    valid, message = validate_guess(guess)

    if not valid:
        return {
            "correct": False,
            "valid": False,
            "message": message,
            "word": guess,
            "feedback": [],
            "answer": answer
        }


    feedback = ["gray"] * len(guess)

    # Track which answer letters are already used
    answer_letters = list(answer)

    # Step 1: Check green letters
    for i in range(len(guess)):

        if guess[i] == answer[i]:

            feedback[i] = "green"

            answer_letters[i] = None


    # Step 2: Check yellow letters
    for i in range(len(guess)):

        if feedback[i] == "green":
            continue


        if guess[i] in answer_letters:

            feedback[i] = "yellow"

            # Remove one occurrence only
            index = answer_letters.index(guess[i])
            answer_letters[index] = None


    return {
        "correct": guess == answer,
        "valid": True,
        "word": guess,
        "feedback": feedback,
        "answer": answer
    }