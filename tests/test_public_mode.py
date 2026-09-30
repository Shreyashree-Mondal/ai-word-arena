"""The safeguards that make a public deployment safe (AIWORD_PUBLIC_MODE=1)."""

import importlib
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import config
from ai import hints, llm_client
from database.models import Puzzle
from game import feedback
from game.topics import CUSTOM_LABEL
from test_app_smoke import (
    APP,
    button,
    expander_labels,
    fresh_app,
    log_out,
    open_app,
    play_mcq_round,
    start_mcq,
)


@pytest.fixture
def public(monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_MODE", True)
    monkeypatch.setattr(config, "AI_SESSION_LIMIT", 15)


@pytest.fixture
def ai_stub(monkeypatch):
    """AI 'is connected' and every call is counted and returns a usable reply."""
    calls = []

    monkeypatch.setattr(llm_client, "is_available", lambda: True)

    def generate(prompt, **kwargs):
        calls.append(prompt)

        if "helping a player" in prompt:                       # a hint request: each reply is different
            return f"Helpful clue number {len(calls)} that never names the word."

        return "Great session! You did well on Space, so try a few more Finance rounds next."

    monkeypatch.setattr(llm_client, "generate_text", generate)

    return calls


# ---------------- the settings ----------------

def reload_config(**env):
    keys = ["AIWORD_PUBLIC_MODE", "AIWORD_AI_SESSION_LIMIT", "AIWORD_AUTOSAVE", "AIWORD_PUZZLE_ORDER"]

    for k in keys:
        os.environ.pop(k, None)

    os.environ.update(env)

    return importlib.reload(config)


def test_public_mode_settings_are_locked_down_and_normal_mode_is_untouched():
    try:
        public = reload_config(AIWORD_PUBLIC_MODE="1")

        assert public.PUBLIC_MODE is True and public.AUTO_SAVE_AI_PUZZLES is False
        assert "ai" not in public.PUZZLE_ORDER and "bank" in public.PUZZLE_ORDER
        assert public.AI_SESSION_LIMIT == 15

        normal = reload_config()

        assert normal.PUBLIC_MODE is False and normal.AUTO_SAVE_AI_PUZZLES is True
        assert normal.PUZZLE_ORDER == ["ai", "bank", "local"]
    finally:
        reload_config()


def test_public_mode_wins_even_if_autosave_or_ai_puzzles_were_requested():
    try:
        cfg = reload_config(AIWORD_PUBLIC_MODE="1", AIWORD_AUTOSAVE="1", AIWORD_PUZZLE_ORDER="ai,bank,local")

        assert cfg.AUTO_SAVE_AI_PUZZLES is False and cfg.PUZZLE_ORDER == ["bank", "local"]

        cfg = reload_config(AIWORD_PUBLIC_MODE="1", AIWORD_PUZZLE_ORDER="ai")

        assert cfg.PUZZLE_ORDER == ["bank", "local"]              # never left with no source at all
    finally:
        reload_config()


def test_the_session_limit_can_be_changed_or_removed():
    try:
        assert reload_config(AIWORD_PUBLIC_MODE="1", AIWORD_AI_SESSION_LIMIT="4").AI_SESSION_LIMIT == 4
        assert reload_config(AIWORD_PUBLIC_MODE="1", AIWORD_AI_SESSION_LIMIT="0").AI_SESSION_LIMIT == 0
    finally:
        reload_config()


# ---------------- what a visitor sees ----------------

def test_public_mode_shows_an_honest_demo_notice_on_the_login_screen(public):
    at = open_app()

    notice = " ".join(i.value for i in at.info)

    assert "public demo" in notice and "reset" in notice and "do not reuse a PIN" in notice


def test_normal_mode_shows_no_demo_notice():
    assert not any("public demo" in i.value for i in open_app().info)


def test_public_mode_has_no_custom_topic_option(public):
    at = fresh_app("Visitor")

    assert CUSTOM_LABEL not in at.selectbox[0].options and len(at.selectbox[0].options) == len(config.SECTORS)


def test_normal_mode_still_offers_custom_topics():
    assert CUSTOM_LABEL in fresh_app("Owner").selectbox[0].options


def test_public_mode_never_offers_or_runs_the_agent_or_any_ai_puzzle_generation(public, ai_stub, monkeypatch):
    from ai import puzzle_manager

    attempts = []                                   # count attempts: a swallowed error would hide a raise

    monkeypatch.setattr(puzzle_manager, "generate_with_agent", lambda *a, **k: attempts.append("agent"))
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: attempts.append("live"))

    at = fresh_app("Visitor")

    assert not any(c.key == "use_agent" for c in at.checkbox)

    at.selectbox[1].select("MCQ")
    button(at, "Start Game").click().run()

    assert attempts == [] and ai_stub == []          # neither generator was even tried, no AI call was made
    assert at.session_state["puzzle"].source in ("bank", "local")


def test_public_mode_refuses_ai_puzzles_even_if_the_puzzle_order_still_allows_them(public, ai_stub, monkeypatch):
    from ai import puzzle_manager

    monkeypatch.setattr(config, "PUZZLE_ORDER", ["ai", "bank", "local"])      # misconfigured on purpose
    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("tried AI")))
    monkeypatch.setattr(puzzle_manager, "generate_with_agent", lambda *a, **k: (_ for _ in ()).throw(AssertionError("tried agent")))

    tried = []
    original = puzzle_manager.SOURCES["ai"]
    monkeypatch.setitem(puzzle_manager.SOURCES, "ai", lambda *a: tried.append(1) or original(*a))

    puzzle = puzzle_manager.get_puzzle("Space", "Easy", "MCQ", use_agent=True)

    assert puzzle.source == "bank"
    assert ai_stub == []                                                     # not a single AI call


def test_the_puzzle_source_is_the_reviewed_bank_even_when_ai_is_connected(monkeypatch, ai_stub):
    monkeypatch.setattr(config, "PUZZLE_ORDER", ["bank", "local"])            # what public mode configures

    from ai import puzzle_manager

    monkeypatch.setattr(puzzle_manager, "generate_live_puzzle", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    assert puzzle_manager.get_puzzle("Space", "Easy", "MCQ").source == "bank"


# ---------------- the per-visitor AI allowance ----------------

def click_hint(at):
    button(at, "Need another hint").click().run()
    assert not at.exception


def sidebar_text(at):
    return " ".join(c.value for c in at.sidebar.caption)


def test_ai_hints_stop_at_the_limit_and_the_game_carries_on_with_built_in_hints(public, ai_stub, monkeypatch):
    monkeypatch.setattr(config, "AI_SESSION_LIMIT", 2)

    at = fresh_app("Visitor")
    start_mcq(at)

    for _ in range(3):
        click_hint(at)

    hints_shown = at.session_state["hints"]

    assert len(ai_stub) == 2                                     # only two AI calls were made
    assert at.session_state["ai_calls"] == 2
    assert "Helpful clue" in hints_shown[0] and "Helpful clue" in hints_shown[1]
    assert "Helpful clue" not in hints_shown[2] and "starts with" in hints_shown[2]   # the built-in hint
    assert "allowance is used up" in sidebar_text(at)


def test_feedback_also_respects_the_allowance(public, ai_stub, monkeypatch):
    monkeypatch.setattr(config, "AI_SESSION_LIMIT", 1)

    at = fresh_app("Visitor")
    start_mcq(at)
    click_hint(at)                                               # uses the single allowed AI call
    play_mcq_round(at)
    button(at, "End session").click().run()

    assert len(ai_stub) == 1
    assert any("Auto-generated feedback" in c.value for c in at.caption)


def test_the_remaining_allowance_is_shown_and_counts_down(public, ai_stub, monkeypatch):
    monkeypatch.setattr(config, "AI_SESSION_LIMIT", 5)

    at = fresh_app("Visitor")

    assert "5 left this session" in sidebar_text(at)

    start_mcq(at)
    click_hint(at)

    assert "4 left this session" in sidebar_text(at)


def test_the_allowance_belongs_to_one_visit_not_to_the_whole_site(public, ai_stub, monkeypatch):
    monkeypatch.setattr(config, "AI_SESSION_LIMIT", 1)

    first = fresh_app("VisitorOne")
    start_mcq(first)
    click_hint(first)

    second = fresh_app("VisitorTwo")                             # a different visitor
    start_mcq(second)
    click_hint(second)

    assert len(ai_stub) == 2                                     # both got their one AI hint


def test_a_limit_of_zero_means_unlimited_in_public_mode(public, ai_stub, monkeypatch):
    monkeypatch.setattr(config, "AI_SESSION_LIMIT", 0)

    at = fresh_app("Visitor")
    start_mcq(at)

    for _ in range(3):
        click_hint(at)

    assert len(ai_stub) == 3 and "left this session" not in sidebar_text(at)


def test_the_limit_does_not_apply_outside_public_mode(ai_stub, monkeypatch):
    monkeypatch.setattr(config, "AI_SESSION_LIMIT", 1)           # ignored: this is not a public site

    at = fresh_app("Owner")
    start_mcq(at)

    for _ in range(3):
        click_hint(at)

    hint_calls = [p for p in ai_stub if "helping a player" in p]

    assert len(hint_calls) == 3                                  # no cap outside public mode


def test_the_allowance_is_reset_by_logging_out_and_in(public, ai_stub, monkeypatch):
    monkeypatch.setattr(config, "AI_SESSION_LIMIT", 1)

    at = fresh_app("Visitor")
    start_mcq(at)
    click_hint(at)
    log_out(at)

    from test_app_smoke import log_in_as

    log_in_as(at, "Visitor")
    start_mcq(at)
    click_hint(at)

    assert len(ai_stub) == 2


def test_an_ai_call_counts_against_the_allowance_even_when_its_reply_is_unusable(public, monkeypatch):
    """Otherwise a run of bad replies would burn real requests without ever reaching the cap."""
    calls = []

    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(config, "AI_SESSION_LIMIT", 2)

    def useless(prompt, **kwargs):
        calls.append(1)
        return "The answer is here"                    # too short/duplicate-ish: rejected or ignored

    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: (calls.append(1) or None))

    at = fresh_app("Visitor")
    start_mcq(at)

    for _ in range(3):
        click_hint(at)

    assert len(calls) == 2                             # the AI returned nothing, yet the third try never happened
    assert at.session_state["ai_calls"] == 2
    assert all("letters" in h or "starts" in h or "ends" in h for h in at.session_state["hints"])


def test_the_hint_function_reports_each_attempt_through_on_call(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)

    puzzle = Puzzle(word="ROCKET", sentence="_____", theme="Space", difficulty="Easy")
    made = []

    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: "A clue that names the ROCKET itself")
    hints.get_extra_hint(puzzle, [], on_call=lambda: made.append("leaky reply"))

    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: None)
    hints.get_extra_hint(puzzle, [], on_call=lambda: made.append("no reply"))

    hints.get_extra_hint(puzzle, [], use_ai=False, on_call=lambda: made.append("AI switched off"))

    assert made == ["leaky reply", "no reply"]          # counted when a request was made, never when it was not


def test_feedback_reports_each_attempt_through_on_call(monkeypatch):
    from database.models import Player

    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: None)

    summary = feedback.build_summary(Player(name="t", games_played=2, correct_answers=1), 1)
    made = []

    feedback.get_session_feedback(summary, on_call=lambda: made.append(1))
    feedback.get_session_feedback(feedback.build_summary(Player(name="t"), 1), on_call=lambda: made.append(1))   # 0 rounds: no call

    assert made == [1]


# ---------------- hints and feedback can be told not to use AI ----------------

def test_use_ai_false_means_no_ai_call_at_all(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no AI call expected")))

    puzzle = Puzzle(word="ROCKET", sentence="_____", theme="Space", difficulty="Easy")

    text, from_ai = hints.get_extra_hint(puzzle, [], use_ai=False)

    assert from_ai is False and "letters" in text

    from database.models import Player

    player = Player(name="t", games_played=2, correct_answers=1)
    summary = feedback.build_summary(player, 1)

    assert feedback.get_session_feedback(summary, use_ai=False)[1] is False


# ---------------- Streamlit secrets reach the settings ----------------

def test_secrets_are_copied_into_the_environment_but_never_override_it(monkeypatch):
    monkeypatch.setenv("AIWORD_BRIDGE_KEEP", "mine")

    at = AppTest.from_file(APP, default_timeout=30)
    at.secrets["AIWORD_BRIDGE_NEW"] = "hello"
    at.secrets["AIWORD_BRIDGE_KEEP"] = "theirs"
    at.secrets["AIWORD_BRIDGE_TABLE"] = {"nested": "value"}          # only plain text secrets are copied
    at.secrets["not a valid name!"] = "x"

    try:
        at.run()

        assert not at.exception
        assert os.environ["AIWORD_BRIDGE_NEW"] == "hello"
        assert os.environ["AIWORD_BRIDGE_KEEP"] == "mine"            # a real environment variable wins
        assert "AIWORD_BRIDGE_TABLE" not in os.environ and "not a valid name!" not in os.environ
    finally:
        os.environ.pop("AIWORD_BRIDGE_NEW", None)


def test_the_app_runs_normally_when_there_are_no_secrets_at_all():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()

    assert not at.exception and at.text_input[0].label == "Your name"
