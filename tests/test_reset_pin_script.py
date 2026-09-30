import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from database import store  # noqa: E402
from reset_pin import main  # noqa: E402


def test_reset_clears_the_pin_keeps_progress_and_lets_the_owner_choose_a_new_one():
    from database.models import Player
    from game.difficulty import DifficultyManager

    store.create_player("Sam", "4827")
    store.save_progress("Sam", Player(name="Sam", total_score=45, games_played=3), DifficultyManager(), [])

    lines = []

    assert main(["sam"], out=lines.append) == 0
    assert "Progress is untouched" in "\n".join(lines)

    back = store.authenticate("Sam", "1357")               # the owner picks a new PIN

    assert back.pin_was_set and back.player.total_score == 45

    with pytest.raises(store.WrongPin):
        store.authenticate("Sam", "4827")                  # the old PIN no longer works


def test_reset_of_an_unknown_player_says_so():
    lines = []

    assert main(["Nobody"], out=lines.append) == 1
    assert "No player called Nobody" in "\n".join(lines)
