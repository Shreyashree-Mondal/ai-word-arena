from ai.puzzle_manager import get_puzzle


def test_wordle_puzzles_are_six_letters():
    for _ in range(20):
        assert len(get_puzzle("Technology", "Easy", "Wordle").word) == 6


def test_mcq_has_shuffled_options_containing_the_answer():
    first_positions = set()

    for _ in range(40):
        p = get_puzzle("Finance", "Medium", "MCQ")

        assert p.word in p.options and len(p.options) == 4
        first_positions.add(p.options.index(p.word))

    assert len(first_positions) > 1  # answer is not always in the same slot


def test_used_words_are_not_repeated_while_fresh_ones_exist():
    used = []

    for _ in range(2):  # 2 Easy puzzles per theme in the starter bank
        p = get_puzzle("History", "Easy", "MCQ", used)
        assert p.word not in used
        used.append(p.word)
