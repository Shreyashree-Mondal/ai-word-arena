"""PIN protection as a player experiences it in the app."""

from datetime import datetime, timedelta, timezone

import pytest
from streamlit.proto.TextInput_pb2 import TextInput

import config
from database import store
from test_app_smoke import (  # helpers, not tests
    PIN,
    button,
    fresh_app,
    log_in_as,
    log_out,
    open_app,
    play_mcq_round,
    sign_up_as,
    start_mcq,
)
from test_store import make_legacy_database


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(store, "_utcnow", c)
    return c


def warnings(at):
    return [w.value for w in at.warning]


def known_player_screen(name="Sam", pin=PIN):
    """A logged-out app where `name` already has a profile."""
    at = fresh_app(name, pin)
    log_out(at)
    return at


# ---------------- the screen ----------------

def test_login_screen_has_two_tabs_guest_option_and_hidden_pin_boxes():
    at = open_app()

    assert [t.label for t in at.tabs] == ["Log in", "New player"]
    assert any("Play as guest" in b.label for b in at.button)

    for key in ["login_pin", "new_pin", "new_pin2"]:
        assert at.text_input(key=key).proto.type == TextInput.PASSWORD      # typed digits are masked

    assert at.text_input(key="login_name").proto.type == TextInput.DEFAULT   # the name is not

    assert "player" not in at.session_state and len(at.selectbox) == 0


# ---------------- creating a profile ----------------

@pytest.mark.parametrize("name,pin,confirm,message", [
    ("Sam", "4827", "4828", "do not match"),
    ("Sam", "1111", "1111", "too easy"),
    ("Sam", "1234", "1234", "too easy"),
    ("Sam", "12", "12", "4 to 8 digits"),
    ("Sam", "12ab", "12ab", "4 to 8 digits"),
    ("Sam", "", "", "4 to 8 digits"),
    ("!!!", "4827", "4827", "Please type a name"),
    ("", "4827", "4827", "Please type a name"),
])
def test_bad_sign_ups_are_refused_and_create_nothing(name, pin, confirm, message):
    at = open_app()

    sign_up_as(at, name, pin, confirm)

    assert any(message in w for w in warnings(at)), warnings(at)
    assert "player" not in at.session_state

    with pytest.raises(store.NoSuchPlayer):
        store.authenticate(name or "Sam", pin or "4827")


def test_signing_up_logs_you_in_and_greets_you():
    at = fresh_app("Sam")

    assert at.session_state["player"].name == "Sam" and at.session_state["persist"] is True
    assert any("Nice to meet you, Sam" in s.value for s in at.success)


def test_a_name_that_is_taken_cannot_be_registered_again_and_the_owner_is_unaffected():
    at = known_player_screen("Sam", PIN)

    sign_up_as(at, "sam", "9351")                                   # different case, different PIN

    assert any("already taken" in w for w in warnings(at)) and "player" not in at.session_state

    log_in_as(at, "Sam", "9351")                                    # the intruder's PIN does not work
    assert "player" not in at.session_state and any("Wrong PIN" in w for w in warnings(at))

    log_in_as(at, "Sam", PIN)                                       # the owner's PIN still does
    assert at.session_state["player"].name == "Sam"


# ---------------- logging in ----------------

def test_correct_pin_restores_progress():
    at = fresh_app("Sam")
    start_mcq(at)
    play_mcq_round(at)
    log_out(at)

    log_in_as(at, "sAM", PIN)

    assert at.session_state["player"].total_score == 10
    assert any("Welcome back, Sam" in s.value for s in at.success)


def test_unknown_name_is_explained_and_nobody_is_logged_in():
    at = open_app()

    log_in_as(at, "Nobody", PIN)

    assert any("There is no player called Nobody" in w for w in warnings(at))
    assert "player" not in at.session_state


def test_login_pin_must_look_like_a_pin():
    at = known_player_screen()

    log_in_as(at, "Sam", "12")

    assert any("4 to 8 digits" in w for w in warnings(at)) and "player" not in at.session_state


def test_wrong_pin_never_logs_in_and_counts_down_the_tries():
    at = known_player_screen()

    seen = []

    for _ in range(4):
        log_in_as(at, "Sam", "0000")
        assert "player" not in at.session_state
        seen.append(next(w for w in warnings(at) if "Wrong PIN" in w))

    assert [w.split(".")[1].strip().split()[0] for w in seen] == ["4", "3", "2", "1"]
    assert "1 more wrong try" in seen[-1]


def test_five_wrong_pins_lock_the_profile_and_even_the_right_pin_is_refused(clock):
    at = known_player_screen()

    for _ in range(config.PIN_MAX_ATTEMPTS):
        log_in_as(at, "Sam", "0000")

    assert any("locked" in w for w in warnings(at)) and "5 minutes" in " ".join(warnings(at))

    log_in_as(at, "Sam", PIN)                                       # correct, but locked out

    assert "player" not in at.session_state and any("locked" in w for w in warnings(at))

    clock.advance(config.PIN_LOCK_SECONDS + 5)

    log_in_as(at, "Sam", PIN)

    assert at.session_state["player"].name == "Sam"


def test_a_locked_profile_does_not_block_other_players(clock):
    at = known_player_screen("Sam", PIN)
    sign_up_as(at, "Ravi", "9351")
    log_out(at)

    for _ in range(config.PIN_MAX_ATTEMPTS):
        log_in_as(at, "Sam", "0000")

    log_in_as(at, "Ravi", "9351")

    assert at.session_state["player"].name == "Ravi"


# ---------------- the PIN is not kept around ----------------

def test_the_pin_is_not_left_in_the_session_after_signing_up_or_logging_in():
    at = fresh_app("Sam", "4827")

    assert "4827" not in repr(at.session_state)

    log_out(at)
    log_in_as(at, "Sam", "4827")

    assert "4827" not in repr(at.session_state)


def test_leaderboard_shows_names_and_scores_only():
    at = fresh_app("Sam")
    start_mcq(at)
    play_mcq_round(at)
    log_out(at)

    text = " ".join(str(t.value) for t in at.table)

    assert "Sam" in text and PIN not in text and "pbkdf2" not in text


# ---------------- profiles from before PINs existed ----------------

def test_an_old_profile_keeps_its_progress_and_gets_a_pin_at_next_login():
    make_legacy_database("Sam", score=45)

    at = open_app()
    log_in_as(at, "Sam", "2468")

    assert at.session_state["player"].total_score == 45
    assert any("created before PINs existed" in s.value for s in at.success)

    log_out(at)
    log_in_as(at, "Sam", "9999")                                     # now protected
    assert "player" not in at.session_state

    log_in_as(at, "Sam", "2468")
    assert at.session_state["player"].total_score == 45


# ---------------- guests ----------------

def test_guests_need_no_pin_and_have_no_account_settings():
    at = open_app()
    button(at, "Play as guest").click().run()

    assert at.session_state["player"].name == "Guest"
    assert not any("Account" in e.label for e in at.expander)


# ---------------- account settings ----------------

def change_pin(at, old, new, new2=None):
    at.text_input(key="cp_old").set_value(old)
    at.text_input(key="cp_new").set_value(new)
    at.text_input(key="cp_new2").set_value(new if new2 is None else new2)
    button(at, "Change PIN").click().run()
    assert not at.exception


def test_account_settings_are_available_to_logged_in_players():
    at = fresh_app("Sam")

    assert any("Account" in e.label for e in at.expander)


def test_changing_the_pin_replaces_the_old_one():
    at = fresh_app("Sam")

    change_pin(at, PIN, "9351")

    assert any("Your PIN was changed" in s.value for s in at.success)

    log_out(at)
    log_in_as(at, "Sam", PIN)
    assert "player" not in at.session_state                          # the old PIN is dead

    log_in_as(at, "Sam", "9351")
    assert at.session_state["player"].name == "Sam"


@pytest.mark.parametrize("old,new,new2,message", [
    ("0000", "9351", "9351", "not your current PIN"),
    ("", "9351", "9351", "current PIN"),
    (PIN, "1111", "1111", "too easy"),
    (PIN, "9351", "9352", "do not match"),
    (PIN, "12", "12", "4 to 8 digits"),
])
def test_bad_pin_changes_are_refused_and_nothing_changes(old, new, new2, message):
    at = fresh_app("Sam")

    change_pin(at, old, new, new2)

    assert any(message in w.value for w in at.sidebar.warning), [w.value for w in at.sidebar.warning]

    log_out(at)
    log_in_as(at, "Sam", PIN)                                        # still the original PIN
    assert at.session_state["player"].name == "Sam"


def delete_account(at, pin, tick=True):
    at.text_input(key="del_pin").set_value(pin)
    at.checkbox(key="del_sure").set_value(tick)
    button(at, "Delete my saved data").click().run()
    assert not at.exception


def test_deleting_removes_everything_and_frees_the_name():
    at = fresh_app("Sam")
    start_mcq(at)
    play_mcq_round(at)

    delete_account(at, PIN)

    assert "player" not in at.session_state
    assert any("Your saved data was deleted" in i.value for i in at.info)
    assert store.leaderboard() == [] and store.recent_rounds("Sam") == []

    log_in_as(at, "Sam", PIN)
    assert any("There is no player called Sam" in w for w in warnings(at))

    sign_up_as(at, "Sam", "9351")                                    # the name can be used again
    assert at.session_state["player"].total_score == 0


def test_deleting_needs_the_box_ticked_and_the_right_pin():
    at = fresh_app("Sam")

    delete_account(at, PIN, tick=False)
    assert any("Tick the box" in w.value for w in at.sidebar.warning) and "player" in at.session_state

    delete_account(at, "0000")
    assert any("Wrong PIN" in w.value for w in at.sidebar.warning) and "player" in at.session_state

    log_out(at)
    log_in_as(at, "Sam", PIN)
    assert at.session_state["player"].name == "Sam"                  # nothing was deleted


def test_wrong_pins_in_account_settings_count_towards_the_lock(clock):
    at = fresh_app("Sam")

    for _ in range(config.PIN_MAX_ATTEMPTS):
        change_pin(at, "0000", "9351")

    assert any("locked" in w.value for w in at.sidebar.warning)

    log_out(at)
    log_in_as(at, "Sam", PIN)

    assert "player" not in at.session_state                          # locked for everyone, including the owner
