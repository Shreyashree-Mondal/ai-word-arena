from database.models import Player
from game.difficulty import DifficultyManager
from game.scoring import calculate_points, update_player_score
from game.wordle_scoring import evaluate_wordle_result


def test_points_and_streak_bonus():
    assert calculate_points(False, 5) == 0
    assert calculate_points(True, 1) == 10
    assert calculate_points(True, 3) == 15  # every 3rd correct answer


def test_update_player_score_tracks_best_streak():
    p = Player(name="t")

    for _ in range(3):
        update_player_score(p, True)

    update_player_score(p, False)
    update_player_score(p, True)

    assert p.best_streak == 3
    assert p.current_streak == 1
    assert p.games_played == 5
    assert p.correct_answers == 4


def test_difficulty_rises_after_three_correct_and_falls_after_two_wrong():
    d = DifficultyManager()

    for _ in range(3):
        d.record_result(True)

    assert d.get_level() == 2

    d.record_result(False)
    d.record_result(False)

    assert d.get_level() == 1


def test_wordle_result_last_attempt_win_does_not_count_as_strong():
    assert evaluate_wordle_result(2, True) is True
    assert evaluate_wordle_result(5, True) is True
    assert evaluate_wordle_result(6, True) is False
    assert evaluate_wordle_result(6, False) is False
