import pytest

import config
from ai import llm_client, puzzle_manager
from database.models import Puzzle


def fake(word, theme="Space"):
    return Puzzle(
        word=word, sentence="_____", theme=theme, difficulty="Easy",
        options=[word, "AAAAAA", "BBBBBB", "CCCCCC"], hint="clue",
    )


@pytest.fixture
def ai_on(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)


def test_ai_is_tried_first_and_labelled(monkeypatch, ai_on):
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: fake("ROCKET"))

    p = puzzle_manager.get_puzzle("Space", "Easy", "MCQ")

    assert p.word == "ROCKET" and p.source == "ai"


def test_ai_failure_falls_back_to_the_bank(monkeypatch, ai_on):
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: None)

    p = puzzle_manager.get_puzzle("Space", "Easy", "MCQ")

    assert p.source == "bank" and p.theme == "Space"


def test_ai_crash_falls_back_to_the_bank(monkeypatch, ai_on):
    def boom(*a, **k):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", boom)

    assert puzzle_manager.get_puzzle("Space", "Easy", "MCQ").source == "bank"


def test_ai_not_connected_goes_straight_to_the_bank(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: False)

    def must_not_run(*a, **k):
        raise AssertionError("AI should not be called when unavailable")

    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", must_not_run)

    assert puzzle_manager.get_puzzle("Finance", "Easy", "MCQ").source == "bank"


def test_bank_failure_falls_back_to_ai_when_bank_first(monkeypatch, ai_on):
    monkeypatch.setattr(config, "PUZZLE_ORDER", ["bank", "ai", "local"])
    monkeypatch.setitem(puzzle_manager.SOURCES, "bank", lambda *a: None)
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: fake("ROCKET"))

    assert puzzle_manager.get_puzzle("Space", "Easy", "MCQ").source == "ai"


def test_everything_fails_except_local(monkeypatch, ai_on):
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: None)
    monkeypatch.setitem(puzzle_manager.SOURCES, "bank", lambda *a: None)

    p = puzzle_manager.get_puzzle("Technology", "Easy", "MCQ")

    assert p.source == "local"


def test_local_is_always_kept_as_last_resort(monkeypatch):
    monkeypatch.setattr(config, "PUZZLE_ORDER", ["ai"])
    monkeypatch.setattr(llm_client, "is_available", lambda: False)

    assert puzzle_manager.get_puzzle("Technology", "Easy", "MCQ").source == "local"


def test_nothing_anywhere_returns_none(monkeypatch):
    for name in puzzle_manager.SOURCES:
        monkeypatch.setitem(puzzle_manager.SOURCES, name, lambda *a: None)

    assert puzzle_manager.get_puzzle("Space", "Easy", "MCQ") is None


def test_wordle_skips_wrong_length_words_from_any_source(monkeypatch, ai_on):
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: fake("GRAVITY"))

    p = puzzle_manager.get_puzzle("Space", "Easy", "Wordle")

    assert len(p.word) == 6 and p.source == "bank"


def test_unknown_source_names_are_ignored(monkeypatch):
    monkeypatch.setattr(config, "PUZZLE_ORDER", ["banana", "bank"])

    assert puzzle_manager.get_puzzle("Space", "Easy", "MCQ").source == "bank"


# ---------------- the agent in the fallback chain ----------------

def agent_puzzle():
    p = fake("TROPHY")
    p.source = "agent"
    return p


def test_agent_is_used_first_when_enabled(monkeypatch, ai_on):
    def must_not_run(*a, **k):
        raise AssertionError("one-shot generator should not run when the agent succeeds")

    monkeypatch.setattr(puzzle_manager, "generate_with_agent", lambda *a, **k: agent_puzzle())
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", must_not_run)

    p = puzzle_manager.get_puzzle("Space", "Easy", "MCQ", use_agent=True)

    assert p.word == "TROPHY" and p.source == "agent"


def test_agent_failure_falls_back_to_the_one_shot_generator(monkeypatch, ai_on):
    monkeypatch.setattr(puzzle_manager, "generate_with_agent", lambda *a, **k: None)
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: fake("ROCKET"))

    p = puzzle_manager.get_puzzle("Space", "Easy", "MCQ", use_agent=True)

    assert p.source == "ai" and p.word == "ROCKET"


def test_agent_and_generator_both_failing_falls_back_to_the_bank(monkeypatch, ai_on):
    monkeypatch.setattr(puzzle_manager, "generate_with_agent", lambda *a, **k: None)
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: None)

    assert puzzle_manager.get_puzzle("Space", "Easy", "MCQ", use_agent=True).source == "bank"


def test_agent_crash_never_breaks_the_game(monkeypatch, ai_on):
    def boom(*a, **k):
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(puzzle_manager, "generate_with_agent", boom)

    assert puzzle_manager.get_puzzle("Space", "Easy", "MCQ", use_agent=True).source == "bank"


def test_agent_is_skipped_when_disabled_or_ai_is_off(monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("agent should not run")

    monkeypatch.setattr(puzzle_manager, "generate_with_agent", must_not_run)
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: fake("ROCKET"))

    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    assert puzzle_manager.get_puzzle("Space", "Easy", "MCQ", use_agent=False).source == "ai"

    monkeypatch.setattr(llm_client, "is_available", lambda: False)
    assert puzzle_manager.get_puzzle("Space", "Easy", "MCQ", use_agent=True).source == "bank"
