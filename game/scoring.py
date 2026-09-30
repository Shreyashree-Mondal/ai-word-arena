from database.models import Player


BASE_POINTS = 10
STREAK_BONUS = 5


def calculate_points(correct: bool, streak: int) -> int:
    """
    Calculate points based on answer and streak.
    """

    if not correct:
        return 0

    points = BASE_POINTS

    if streak > 0 and streak % 3 == 0:
        points += STREAK_BONUS

    return points



def update_player_score(player: Player, correct: bool) -> int:
    """
    Update player's score and streak.
    """

    if correct:
        player.current_streak += 1

        points = calculate_points(
            correct=True,
            streak=player.current_streak
        )

        player.total_score += points
        player.correct_answers += 1
        player.best_streak = max(player.best_streak, player.current_streak)

    else:
        player.current_streak = 0
        points = 0

    player.games_played += 1

    return points