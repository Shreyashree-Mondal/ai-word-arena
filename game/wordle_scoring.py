def evaluate_wordle_result(
    guesses_used,
    won
):

    if not won:
        return False


    if guesses_used <= 2:
        return True

    if guesses_used <= 5:
        return True

    # solved on last attempt
    return False