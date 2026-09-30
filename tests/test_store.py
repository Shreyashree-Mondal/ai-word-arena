import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

import config
from database import store
from database.models import Player
from game.difficulty import DifficultyManager

PIN = "4827"


def played_player(name="Sam"):
    p = Player(name=name, total_score=45, current_streak=2, best_streak=4, games_played=7,
               correct_answers=5, hints_used=3, favorite_theme="Space")
    p.theme_stats = {"Space": {"played": 4, "correct": 3}, "Music": {"played": 3, "correct": 2}}

    d = DifficultyManager()
    d.level, d.correct_streak, d.wrong_count = 2, 1, 1

    return p, d


def sql(statement, *params):
    conn = sqlite3.connect(config.DB_PATH)
    rows = conn.execute(statement, params).fetchall()
    conn.commit()
    conn.close()
    return rows


class Clock:
    """A controllable replacement for store._utcnow."""

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


# ---------------- creating and signing in ----------------

def test_new_player_is_created_with_defaults():
    loaded = store.create_player("Sam", PIN)

    assert loaded.is_new and loaded.player.name == "Sam" and not loaded.pin_was_set
    assert loaded.player.total_score == 0 and loaded.difficulty.level == 1 and loaded.used_words == []


def test_a_name_can_only_be_taken_once_ignoring_letter_case_and_spacing():
    store.create_player("Sam Lee", PIN)

    for again in ["Sam Lee", "sam lee", "SAM  LEE", "  sam   lee "]:
        with pytest.raises(store.NameTaken):
            store.create_player(again, "9351")

    assert len(store.leaderboard()) == 0                 # and nothing was overwritten
    assert store.authenticate("sam lee", PIN).player.name == "Sam Lee"


def test_right_pin_signs_in_ignoring_letter_case_and_keeps_the_original_spelling():
    store.create_player("Sam", PIN)

    again = store.authenticate("  sAM ", PIN)

    assert again.is_new is False and again.player.name == "Sam"


def test_unknown_name_is_reported_and_creates_nothing():
    with pytest.raises(store.NoSuchPlayer):
        store.authenticate("Nobody", PIN)

    with pytest.raises(store.NoSuchPlayer):
        store.authenticate("", PIN)

    assert sql("SELECT COUNT(*) FROM players") == [(0,)]


def test_blank_name_cannot_be_created():
    with pytest.raises(ValueError):
        store.create_player("   ", PIN)


# ---------------- the PIN is protected ----------------

def test_the_pin_is_never_stored_as_digits_and_each_hash_is_salted():
    store.create_player("Sam", PIN)
    store.create_player("Ravi", PIN)                      # same PIN, different player

    hashes = [r[0] for r in sql("SELECT pin_hash FROM players ORDER BY key")]

    dump = " ".join(str(v) for row in sql("SELECT * FROM players") for v in row)

    assert PIN not in hashes[0].split("$")[0] and hashes[0].startswith("pbkdf2_sha256$1000$")
    assert hashes[0] != hashes[1]                         # random salt: same PIN, different hash
    assert f"${PIN}$" not in dump and PIN not in hashes[0] + hashes[1]


def test_iteration_count_comes_from_settings_and_is_stored_with_the_hash(monkeypatch):
    monkeypatch.setattr(config, "PIN_HASH_ITERATIONS", 1234)

    store.create_player("Sam", PIN)

    assert sql("SELECT pin_hash FROM players")[0][0].startswith("pbkdf2_sha256$1234$")

    monkeypatch.setattr(config, "PIN_HASH_ITERATIONS", 2000)   # later change must not break old hashes

    assert store.authenticate("Sam", PIN).player.name == "Sam"


def test_wrong_pin_is_rejected_and_counts_down_the_remaining_tries():
    store.create_player("Sam", PIN)

    left = []

    for _ in range(4):
        with pytest.raises(store.WrongPin) as err:
            store.authenticate("Sam", "0000")

        left.append(err.value.attempts_left)

    assert left == [4, 3, 2, 1]


def test_five_wrong_pins_lock_the_profile_even_against_the_correct_pin(clock):
    store.create_player("Sam", PIN)

    for _ in range(config.PIN_MAX_ATTEMPTS - 1):
        with pytest.raises(store.WrongPin):
            store.authenticate("Sam", "0000")

    with pytest.raises(store.Locked) as err:
        store.authenticate("Sam", "0000")                 # the 5th wrong try

    assert err.value.seconds == config.PIN_LOCK_SECONDS

    clock.advance(60)

    with pytest.raises(store.Locked) as err:              # the CORRECT pin is refused while locked
        store.authenticate("Sam", PIN)

    assert 200 < err.value.seconds <= config.PIN_LOCK_SECONDS - 59

    clock.advance(config.PIN_LOCK_SECONDS)                # lock expires

    assert store.authenticate("Sam", PIN).player.name == "Sam"


def test_after_a_lock_expires_the_counter_starts_from_scratch(clock):
    store.create_player("Sam", PIN)

    for _ in range(config.PIN_MAX_ATTEMPTS):
        with pytest.raises(store.AuthError):
            store.authenticate("Sam", "0000")

    clock.advance(config.PIN_LOCK_SECONDS + 1)

    with pytest.raises(store.WrongPin) as err:
        store.authenticate("Sam", "0000")

    assert err.value.attempts_left == config.PIN_MAX_ATTEMPTS - 1


def test_a_correct_pin_resets_the_wrong_try_counter():
    store.create_player("Sam", PIN)

    for _ in range(3):
        with pytest.raises(store.WrongPin):
            store.authenticate("Sam", "0000")

    store.authenticate("Sam", PIN)

    for expected in [4, 3]:
        with pytest.raises(store.WrongPin) as err:
            store.authenticate("Sam", "0000")

        assert err.value.attempts_left == expected


def test_locking_one_player_does_not_affect_another(clock):
    store.create_player("Sam", PIN)
    store.create_player("Ravi", "9351")

    for _ in range(config.PIN_MAX_ATTEMPTS):
        with pytest.raises(store.AuthError):
            store.authenticate("Sam", "0000")

    assert store.authenticate("Ravi", "9351").player.name == "Ravi"


@pytest.mark.parametrize("garbage", [None, "", "not a hash", "a$b$c$d", "pbkdf2_sha256$x$zz$yy",
                                     "pbkdf2_sha256$99999999999$00$00", "md5$1$00$00", "pbkdf2_sha256$0$00$00"])
def test_a_damaged_stored_hash_fails_safely(garbage):
    store.create_player("Sam", PIN)
    sql("UPDATE players SET pin_hash = ? WHERE key = 'sam'", garbage)

    if garbage:
        with pytest.raises(store.WrongPin):               # never a crash, never a successful login
            store.authenticate("Sam", PIN)


# ---------------- profiles created before PINs existed ----------------

def make_legacy_database(name="Sam", score=45):
    """A database exactly as the earlier, PIN-less version created it."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.executescript("""
        CREATE TABLE players (key TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL, total_score INTEGER NOT NULL DEFAULT 0,
            games_played INTEGER NOT NULL DEFAULT 0, correct_answers INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL);
        CREATE TABLE rounds (id INTEGER PRIMARY KEY AUTOINCREMENT, player_key TEXT NOT NULL, ts TEXT NOT NULL,
            mode TEXT, theme TEXT, difficulty TEXT, word TEXT, correct INTEGER NOT NULL, points INTEGER NOT NULL,
            hints INTEGER NOT NULL DEFAULT 0, source TEXT);
    """)
    state = json.dumps({"total_score": score, "games_played": 3, "correct_answers": 2})
    conn.execute("INSERT INTO players VALUES (?,?,?,?,?,?,?,?)",
                 (name.casefold(), name, "2026-09-01", "2026-09-01", score, 3, 2, state))
    conn.execute("INSERT INTO rounds (player_key, ts, correct, points, word) VALUES (?, 'x', 1, 10, 'ROCKET')",
                 (name.casefold(),))
    conn.commit()
    conn.close()


def test_an_old_database_is_upgraded_in_place_without_losing_anything():
    make_legacy_database()

    loaded = store.authenticate("Sam", "2468")

    assert loaded.pin_was_set is True
    assert loaded.player.total_score == 45 and loaded.player.games_played == 3
    assert [r["word"] for r in store.recent_rounds("Sam")] == ["ROCKET"]


def test_first_login_of_an_old_profile_sets_the_pin_and_protects_it_from_then_on():
    make_legacy_database()

    store.authenticate("Sam", "2468")

    with pytest.raises(store.WrongPin):
        store.authenticate("Sam", "9999")

    again = store.authenticate("sam", "2468")

    assert again.pin_was_set is False and again.player.total_score == 45


def test_signing_up_cannot_take_over_an_old_profile(clock):
    make_legacy_database()

    with pytest.raises(store.NameTaken):
        store.create_player("Sam", "2468")                # signing up cannot take over an old profile


# ---------------- change / reset / delete ----------------

def test_change_pin_needs_the_current_pin_and_replaces_it():
    store.create_player("Sam", PIN)

    with pytest.raises(store.WrongPin):
        store.change_pin("Sam", "0000", "9351")

    store.change_pin("Sam", PIN, "9351")

    with pytest.raises(store.WrongPin):
        store.authenticate("Sam", PIN)

    assert store.authenticate("Sam", "9351").player.name == "Sam"


def test_wrong_current_pins_in_change_pin_count_towards_the_lock(clock):
    store.create_player("Sam", PIN)

    for _ in range(config.PIN_MAX_ATTEMPTS - 1):
        with pytest.raises(store.WrongPin):
            store.change_pin("Sam", "0000", "9351")

    with pytest.raises(store.Locked):
        store.change_pin("Sam", "0000", "9351")


def test_reset_pin_lets_the_owner_choose_a_new_one_and_clears_a_lock(clock):
    store.create_player("Sam", PIN)
    store.save_progress("Sam", *played_player()[:2], [])

    for _ in range(config.PIN_MAX_ATTEMPTS):
        with pytest.raises(store.AuthError):
            store.authenticate("Sam", "0000")

    assert store.reset_pin("sam") is True
    assert store.reset_pin("nobody") is False

    back = store.authenticate("Sam", "1357")

    assert back.pin_was_set and back.player.total_score == 45     # progress kept
    assert store.authenticate("Sam", "1357").pin_was_set is False


def test_delete_needs_the_pin_and_removes_everything_about_that_player_only():
    store.create_player("Sam", PIN)
    store.create_player("Ravi", "9351")
    store.log_round("Sam", "MCQ", "Space", "Easy", "ROCKET", True, 10)
    store.log_round("Ravi", "MCQ", "Space", "Easy", "PLANET", True, 10)

    with pytest.raises(store.WrongPin):
        store.delete_player("Sam", "0000")

    assert len(store.recent_rounds("Sam")) == 1               # nothing happened

    store.delete_player("sam", PIN)

    assert sql("SELECT COUNT(*) FROM players WHERE key = 'sam'") == [(0,)]
    assert store.recent_rounds("Sam") == [] and len(store.recent_rounds("Ravi")) == 1
    assert store.authenticate("Ravi", "9351").player.name == "Ravi"

    with pytest.raises(store.NoSuchPlayer):
        store.authenticate("Sam", PIN)

    store.create_player("Sam", "2468")                         # the name is free again


# ---------------- saving progress ----------------

def test_everything_survives_a_save_and_login():
    p, d = played_player()

    store.create_player("Sam", PIN)
    store.save_progress("Sam", p, d, ["ROCKET", "PLANET"])

    back = store.authenticate("sam", PIN)

    assert back.player.total_score == 45 and back.player.current_streak == 2
    assert back.player.best_streak == 4 and back.player.games_played == 7
    assert back.player.correct_answers == 5 and back.player.hints_used == 3
    assert back.player.theme_stats == p.theme_stats
    assert (back.difficulty.level, back.difficulty.correct_streak, back.difficulty.wrong_count) == (2, 1, 1)
    assert back.used_words == ["ROCKET", "PLANET"]


def test_players_do_not_see_each_others_progress():
    p, d = played_player("Sam")

    store.create_player("Sam", PIN)
    store.save_progress("Sam", p, d, [])

    other = store.create_player("Ravi", "9351")

    assert other.player.total_score == 0 and other.player.theme_stats == {}


def test_saving_never_creates_a_profile_because_creating_needs_a_pin():
    p, d = played_player("Ghost")

    store.save_progress("Ghost", p, d, [])

    assert sql("SELECT COUNT(*) FROM players") == [(0,)]


def test_saving_does_not_disturb_the_pin_or_lock_state():
    store.create_player("Sam", PIN)

    for _ in range(2):
        with pytest.raises(store.WrongPin):
            store.authenticate("Sam", "0000")

    store.save_progress("Sam", *played_player()[:2], [])

    assert sql("SELECT failed_attempts FROM players") == [(2,)]
    assert store.authenticate("Sam", PIN).player.total_score == 45


def test_used_words_are_capped_keeping_the_most_recent():
    p, d = played_player()
    words = [f"W{i}" for i in range(store.MAX_USED_WORDS + 25)]

    store.create_player("Sam", PIN)
    store.save_progress("Sam", p, d, words)

    kept = store.authenticate("Sam", PIN).used_words

    assert len(kept) == store.MAX_USED_WORDS and kept[-1] == words[-1] and kept[0] == words[25]


@pytest.mark.parametrize("raw", ["", "{not json", "null", "[1, 2]", '"text"', "42"])
def test_damaged_saved_state_falls_back_to_defaults_instead_of_crashing(raw):
    store.create_player("Sam", PIN)
    sql("UPDATE players SET state = ? WHERE key = 'sam'", raw)

    loaded = store.authenticate("Sam", PIN)

    assert loaded.player.total_score == 0 and loaded.difficulty.level == 1 and loaded.used_words == []


def test_absurd_saved_values_are_clamped_and_wrong_types_ignored():
    store.create_player("Sam", PIN)

    evil = json.dumps({
        "total_score": -50, "games_played": "many", "best_streak": 10**30,
        "difficulty": {"level": 99, "correct_streak": 50, "wrong_count": -3},
        "theme_stats": {"Space": "oops", "Music": {"played": "x", "correct": 2}, "7": None},
        "used_words": "not a list", "favorite_theme": "x" * 500,
    })
    sql("UPDATE players SET state = ? WHERE key = 'sam'", evil)

    loaded = store.authenticate("Sam", PIN)

    assert loaded.player.total_score == 0 and loaded.player.games_played == 0
    assert loaded.player.best_streak <= 10**9 and len(loaded.player.favorite_theme) <= 60
    assert loaded.difficulty.level == 3 and loaded.difficulty.correct_streak == 2 and loaded.difficulty.wrong_count == 0
    assert loaded.player.theme_stats == {"Music": {"played": 0, "correct": 2}}
    assert loaded.used_words == []


# ---------------- round history ----------------

def log(name="Sam", word="ROCKET", correct=True, points=10, **kw):
    store.log_round(name, "MCQ", "Space", "Easy", word, correct, points, **kw)


def test_recent_rounds_are_newest_first_limited_and_per_player():
    for i, w in enumerate(["AAAAAA", "BBBBBB", "CCCCCC", "DDDDDD"]):
        log(word=w, correct=i % 2 == 0, points=i)

    log(name="Ravi", word="ZZZZZZ")

    rounds = store.recent_rounds("sam", limit=3)

    assert [r["word"] for r in rounds] == ["DDDDDD", "CCCCCC", "BBBBBB"]
    assert rounds[0]["correct"] == 0 and rounds[1]["correct"] == 1
    assert [r["word"] for r in store.recent_rounds("Ravi")] == ["ZZZZZZ"]
    assert store.recent_rounds("nobody") == []


def test_round_details_are_stored():
    log(hints=2, source="agent", points=15)

    r = store.recent_rounds("Sam")[0]

    assert (r["hints"], r["source"], r["points"], r["difficulty"]) == (2, "agent", 15, "Easy")


def test_hostile_text_is_stored_as_plain_data_never_run_as_sql():
    nasty = "Robert'); DROP TABLE players;--"

    store.create_player(nasty, PIN)
    store.log_round(nasty, "MCQ", nasty, "Easy", "ROCKET", True, 10)

    assert store.authenticate(nasty, PIN).is_new is False
    assert store.recent_rounds(nasty)[0]["theme"] == nasty
    assert isinstance(store.leaderboard(), list)               # the tables still exist

    with pytest.raises(store.WrongPin):
        store.authenticate(nasty, "' OR '1'='1")               # injection in the PIN field


# ---------------- leaderboard ----------------

def test_leaderboard_orders_by_score_hides_players_with_no_rounds_and_computes_accuracy():
    def add(name, score, played, correct):
        store.create_player(name, PIN)
        store.save_progress(name, Player(name=name, total_score=score, games_played=played,
                                         correct_answers=correct), DifficultyManager(), [])

    add("Low", 10, 2, 1)
    add("High", 90, 4, 3)
    add("Mid", 40, 3, 3)
    add("Idle", 0, 0, 0)

    board = store.leaderboard()

    assert [r["name"] for r in board] == ["High", "Mid", "Low"]
    assert [r["accuracy"] for r in board] == [75, 100, 50]
    assert board[0] == {"name": "High", "score": 90, "rounds": 4, "accuracy": 75}
    assert len(store.leaderboard(limit=2)) == 2


def test_the_leaderboard_never_exposes_pins_or_hashes():
    store.create_player("Sam", PIN)
    store.save_progress("Sam", *played_player()[:2], [])

    assert set(store.leaderboard()[0]) == {"name", "score", "rounds", "accuracy"}


# ---------------- robustness ----------------

def test_missing_folders_are_created(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "deep" / "er" / "game.db"))

    assert store.create_player("Sam", PIN).is_new
    assert (tmp_path / "deep" / "er" / "game.db").exists()


def test_many_sessions_working_at_once_do_not_lose_or_corrupt_data():
    errors = []

    def worker(i):
        try:
            store.create_player(f"P{i}", PIN)
            p = Player(name=f"P{i}", total_score=i, games_played=1, correct_answers=1)

            for _ in range(5):
                store.save_progress(f"P{i}", p, DifficultyManager(), [])
                store.log_round(f"P{i}", "MCQ", "Space", "Easy", "ROCKET", True, i)
                store.authenticate(f"P{i}", PIN)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert errors == []
    assert len(store.leaderboard(limit=20)) == 8
    assert all(len(store.recent_rounds(f"P{i}", 50)) == 5 for i in range(8))


def test_only_one_of_two_simultaneous_signups_for_the_same_name_wins():
    results = []

    def worker(pin):
        try:
            store.create_player("Sam", pin)
            results.append("created")
        except store.NameTaken:
            results.append("taken")

    threads = [threading.Thread(target=worker, args=(p,)) for p in ("4827", "9351", "2468", "1357")]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert sorted(results) == ["created", "taken", "taken", "taken"]
    assert sql("SELECT COUNT(*) FROM players") == [(1,)]
