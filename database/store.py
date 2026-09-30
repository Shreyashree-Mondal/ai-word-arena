"""
Saved player profiles and round history (SQLite, standard library only).

A profile is found by NAME (letter case ignored: "Sam" and "sam" are the same
player) and protected by a numeric PIN.

Security model - what the PIN does and does not do:
  * The PIN is stored only as a salted PBKDF2-HMAC-SHA256 hash, never as digits.
  * Wrong PINs are counted per profile; after PIN_MAX_ATTEMPTS the profile is
    locked for PIN_LOCK_SECONDS, even against the correct PIN, so a short PIN
    cannot be brute-forced through the game.
  * It does NOT protect against someone who copies the database FILE: a 4-8
    digit PIN can be brute-forced offline. Protect the file itself if that matters.
  * There is no email, so a forgotten PIN cannot be recovered by the player;
    whoever runs the game can clear it with scripts/reset_pin.py.

Guests are never stored. Every function opens its own short-lived connection,
uses parameterised SQL only, and is safe to call from several sessions at once.
"""

import hashlib
import hmac
import json
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List

import config
from database.models import Player
from game.difficulty import DifficultyManager

MAX_USED_WORDS = 100

_write_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    key             TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    total_score     INTEGER NOT NULL DEFAULT 0,
    games_played    INTEGER NOT NULL DEFAULT 0,
    correct_answers INTEGER NOT NULL DEFAULT 0,
    state           TEXT NOT NULL,
    pin_hash        TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT
);
CREATE TABLE IF NOT EXISTS rounds (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    player_key TEXT NOT NULL,
    ts         TEXT NOT NULL,
    mode       TEXT,
    theme      TEXT,
    difficulty TEXT,
    word       TEXT,
    correct    INTEGER NOT NULL,
    points     INTEGER NOT NULL,
    hints      INTEGER NOT NULL DEFAULT 0,
    source     TEXT
);
CREATE INDEX IF NOT EXISTS rounds_by_player ON rounds (player_key, id);
"""

# Columns added after the first release; older databases are upgraded in place.
_MIGRATIONS = {
    "pin_hash": "ALTER TABLE players ADD COLUMN pin_hash TEXT",
    "failed_attempts": "ALTER TABLE players ADD COLUMN failed_attempts INTEGER NOT NULL DEFAULT 0",
    "locked_until": "ALTER TABLE players ADD COLUMN locked_until TEXT",
}


# ---------------------------------------
# Errors raised for sign-in problems
# ---------------------------------------

class AuthError(Exception):
    """Base class for every sign-in problem."""


class NameTaken(AuthError):
    pass


class NoSuchPlayer(AuthError):
    pass


class WrongPin(AuthError):
    def __init__(self, attempts_left):
        super().__init__(f"Wrong PIN ({attempts_left} tries left)")
        self.attempts_left = attempts_left


class Locked(AuthError):
    def __init__(self, seconds):
        super().__init__(f"Locked for {seconds} more seconds")
        self.seconds = seconds


@dataclass
class LoadedPlayer:
    player: Player
    difficulty: DifficultyManager
    used_words: List[str] = field(default_factory=list)
    is_new: bool = False
    pin_was_set: bool = False   # an old profile without a PIN just received one


# ---------------------------------------
# Helpers
# ---------------------------------------

def _utcnow():
    return datetime.now(timezone.utc)


def _now():
    return _utcnow().isoformat(timespec="seconds")


def _key(name):
    return " ".join((name or "").split()).casefold()


def _connect():
    path = config.DB_PATH

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)

    have = {r["name"] for r in conn.execute("PRAGMA table_info(players)")}

    for column, statement in _MIGRATIONS.items():
        if column not in have:
            conn.execute(statement)

    conn.commit()

    return conn


# ---------------------------------------
# PIN hashing
# ---------------------------------------

def _hash_pin(pin, salt=None, iterations=None):
    salt = salt if salt is not None else os.urandom(16)
    iterations = int(iterations or config.PIN_HASH_ITERATIONS)

    digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations)

    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _check_pin(pin, stored):
    """Constant-time check. Any malformed stored value simply fails."""
    try:
        scheme, iterations, salt_hex, hash_hex = stored.split("$")

        if scheme != "pbkdf2_sha256":
            return False

        iterations = int(iterations)

        if not 1 <= iterations <= 10_000_000:
            return False

        expected = _hash_pin(pin, bytes.fromhex(salt_hex), iterations).split("$")[3]
    except (AttributeError, ValueError):
        return False

    return hmac.compare_digest(expected, hash_hex)


# ---------------------------------------
# (De)serialising a player's state - tolerant of missing or damaged data
# ---------------------------------------

def _int(value, default=0, low=0, high=10**9):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _dump_state(player, difficulty, used_words):
    return json.dumps(
        {
            "total_score": player.total_score,
            "current_streak": player.current_streak,
            "best_streak": player.best_streak,
            "games_played": player.games_played,
            "correct_answers": player.correct_answers,
            "hints_used": player.hints_used,
            "favorite_theme": player.favorite_theme,
            "theme_stats": player.theme_stats,
            "difficulty": {
                "level": difficulty.level,
                "correct_streak": difficulty.correct_streak,
                "wrong_count": difficulty.wrong_count,
            },
            "used_words": list(used_words)[-MAX_USED_WORDS:],
        }
    )


def _load_state(name, raw):
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        data = {}

    if not isinstance(data, dict):
        data = {}

    player = Player(
        name=name,
        total_score=_int(data.get("total_score")),
        current_streak=_int(data.get("current_streak")),
        best_streak=_int(data.get("best_streak")),
        games_played=_int(data.get("games_played")),
        correct_answers=_int(data.get("correct_answers")),
        hints_used=_int(data.get("hints_used")),
        favorite_theme=str(data.get("favorite_theme", ""))[:60],
    )

    stats = data.get("theme_stats")

    if isinstance(stats, dict):
        for theme, s in stats.items():
            if isinstance(s, dict):
                player.theme_stats[str(theme)] = {
                    "played": _int(s.get("played")),
                    "correct": _int(s.get("correct")),
                }

    difficulty = DifficultyManager()
    d = data.get("difficulty")

    if isinstance(d, dict):
        difficulty.level = _int(d.get("level"), 1, 1, 3)
        difficulty.correct_streak = _int(d.get("correct_streak"), 0, 0, 2)
        difficulty.wrong_count = _int(d.get("wrong_count"), 0, 0, 1)

    used = data.get("used_words")
    used = [str(w) for w in used][-MAX_USED_WORDS:] if isinstance(used, list) else []

    return player, difficulty, used


# ---------------------------------------
# Sign-in
# ---------------------------------------

def _verify(conn, key, pin):
    """
    Checks a PIN against a profile and updates the failed-attempt counter.
    Runs inside the caller's transaction and RETURNS the outcome (raising
    inside the transaction would roll the counter back).

    Returns (outcome, row, extra) where outcome is one of:
    "none", "locked", "legacy", "ok", "wrong", "just_locked".
    """
    row = conn.execute(
        "SELECT name, state, pin_hash, failed_attempts, locked_until "
        "FROM players WHERE key = ?",
        (key,),
    ).fetchone()

    if row is None:
        return "none", None, None

    now = _utcnow()

    if row["locked_until"]:
        try:
            until = datetime.fromisoformat(row["locked_until"])
        except ValueError:
            until = now

        if until > now:
            return "locked", row, max(1, int((until - now).total_seconds()) + 1)

    if not row["pin_hash"]:
        return "legacy", row, None

    if _check_pin(pin, row["pin_hash"]):
        conn.execute(
            "UPDATE players SET failed_attempts = 0, locked_until = NULL WHERE key = ?",
            (key,),
        )
        return "ok", row, None

    failed = row["failed_attempts"] + 1

    if failed >= config.PIN_MAX_ATTEMPTS:
        until = (now + timedelta(seconds=config.PIN_LOCK_SECONDS)).isoformat(timespec="seconds")

        conn.execute(
            "UPDATE players SET failed_attempts = 0, locked_until = ? WHERE key = ?",
            (until, key),
        )
        return "just_locked", row, config.PIN_LOCK_SECONDS

    conn.execute("UPDATE players SET failed_attempts = ? WHERE key = ?", (failed, key))

    return "wrong", row, config.PIN_MAX_ATTEMPTS - failed


def _raise_for(outcome, extra):
    if outcome == "none":
        raise NoSuchPlayer("No player with that name.")

    if outcome in ("locked", "just_locked"):
        raise Locked(extra)

    if outcome == "wrong":
        raise WrongPin(extra)


def create_player(name, pin):
    """Creates a new profile protected by `pin`. Raises NameTaken if the name is in use."""
    key = _key(name)

    if not key:
        raise ValueError("A player name is required.")

    with _write_lock, closing(_connect()) as conn:
        with conn:
            if conn.execute("SELECT 1 FROM players WHERE key = ?", (key,)).fetchone():
                taken = True
            else:
                taken = False

                player = Player(name=" ".join(name.split()))
                difficulty = DifficultyManager()
                now = _now()

                conn.execute(
                    "INSERT INTO players (key, name, created_at, last_seen, state, pin_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key, player.name, now, now, _dump_state(player, difficulty, []), _hash_pin(pin)),
                )

    if taken:
        raise NameTaken("That name is already taken.")

    return LoadedPlayer(player, difficulty, [], is_new=True)


def authenticate(name, pin):
    """
    Signs a player in. Raises NoSuchPlayer, WrongPin or Locked.

    A profile created before PINs existed has none: the PIN typed at its next
    login becomes its PIN (pin_was_set=True on the result).
    """
    key = _key(name)

    if not key:
        raise NoSuchPlayer("No player with that name.")

    with _write_lock, closing(_connect()) as conn:
        with conn:
            outcome, row, extra = _verify(conn, key, pin)

            if outcome in ("ok", "legacy"):
                if outcome == "legacy":
                    conn.execute(
                        "UPDATE players SET pin_hash = ?, failed_attempts = 0, "
                        "locked_until = NULL WHERE key = ?",
                        (_hash_pin(pin), key),
                    )

                conn.execute("UPDATE players SET last_seen = ? WHERE key = ?", (_now(), key))

    _raise_for(outcome, extra)

    player, difficulty, used = _load_state(row["name"], row["state"])

    return LoadedPlayer(player, difficulty, used, is_new=False, pin_was_set=(outcome == "legacy"))


def change_pin(name, old_pin, new_pin):
    """Sets a new PIN after checking the current one (wrong tries count towards the lock)."""
    key = _key(name)

    with _write_lock, closing(_connect()) as conn:
        with conn:
            outcome, _row, extra = _verify(conn, key, old_pin)

            if outcome == "ok":
                conn.execute(
                    "UPDATE players SET pin_hash = ? WHERE key = ?", (_hash_pin(new_pin), key)
                )

    if outcome != "ok":
        _raise_for(outcome if outcome != "legacy" else "none", extra)


def reset_pin(name):
    """
    Administrator recovery: removes the PIN and any lock so the owner can set a
    new one at their next login. Returns False if there is no such player.
    """
    with _write_lock, closing(_connect()) as conn:
        with conn:
            cursor = conn.execute(
                "UPDATE players SET pin_hash = NULL, failed_attempts = 0, locked_until = NULL "
                "WHERE key = ?",
                (_key(name),),
            )

    return cursor.rowcount > 0


def delete_player(name, pin):
    """Permanently deletes the profile and its round history (PIN required)."""
    key = _key(name)

    with _write_lock, closing(_connect()) as conn:
        with conn:
            outcome, _row, extra = _verify(conn, key, pin)

            if outcome == "ok":
                conn.execute("DELETE FROM rounds WHERE player_key = ?", (key,))
                conn.execute("DELETE FROM players WHERE key = ?", (key,))

    if outcome != "ok":
        _raise_for(outcome if outcome != "legacy" else "none", extra)


# ---------------------------------------
# Saving progress (only ever called for a signed-in player)
# ---------------------------------------

def save_progress(name, player, difficulty, used_words):
    """Updates an EXISTING profile; never creates one (creation needs a PIN)."""
    key = _key(name)

    if not key:
        return

    with _write_lock, closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE players SET last_seen = ?, total_score = ?, games_played = ?, "
            "correct_answers = ?, state = ? WHERE key = ?",
            (
                _now(),
                player.total_score,
                player.games_played,
                player.correct_answers,
                _dump_state(player, difficulty, used_words),
                key,
            ),
        )


def log_round(name, mode, theme, difficulty, word, correct, points, hints=0, source=""):
    key = _key(name)

    if not key:
        return

    with _write_lock, closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO rounds (player_key, ts, mode, theme, difficulty, word, "
            "correct, points, hints, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key, _now(), mode, theme, difficulty, word,
                int(bool(correct)), int(points), int(hints), source,
            ),
        )


def recent_rounds(name, limit=10):
    """Newest first."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT ts, mode, theme, difficulty, word, correct, points, hints, source "
            "FROM rounds WHERE player_key = ? ORDER BY id DESC LIMIT ?",
            (_key(name), int(limit)),
        ).fetchall()

    return [dict(r) for r in rows]


def leaderboard(limit=10):
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT name, total_score, games_played, correct_answers FROM players "
            "WHERE games_played > 0 "
            "ORDER BY total_score DESC, games_played ASC, name ASC LIMIT ?",
            (int(limit),),
        ).fetchall()

    return [
        {
            "name": r["name"],
            "score": r["total_score"],
            "rounds": r["games_played"],
            "accuracy": round(100 * r["correct_answers"] / r["games_played"]),
        }
        for r in rows
    ]
