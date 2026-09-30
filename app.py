import math
import os
import random

import streamlit as st


def _load_secrets_into_environment():
    """
    On hosted platforms the API key lives in Streamlit's "secrets", not in the environment.
    Copy plain text secrets into the environment (never overriding a variable that is already
    set) BEFORE any project module reads its settings. Does nothing when there are no secrets.
    """
    try:
        secrets = dict(st.secrets)
    except Exception:
        return

    for key, value in secrets.items():
        if isinstance(value, str) and key.isidentifier() and key not in os.environ:
            os.environ[key] = value


_load_secrets_into_environment()


import config
from agent import log as agent_log
from ai import llm_client
from ai.hints import get_extra_hint
from ai.puzzle_manager import get_puzzle
from config import (
    DIFFICULTY_LEVELS,
    MAX_ANAGRAM_ATTEMPTS,
    MAX_HINTS,
    MAX_WORDLE_ATTEMPTS,
    SECTORS,
    WORD_LENGTH,
)
from database import store
from database.models import Player
from game.anagram import check_anagram, scramble
from game.context_quiz import check_answer, check_wordle_answer
from game.difficulty import DifficultyManager
from game.players import GUEST_NAME, clean_name, clean_pin, pin_problem
from game.feedback import build_summary, get_session_feedback, record_round
from game.scoring import update_player_score
from game.topics import CUSTOM_LABEL, clean_topic, has_offline_puzzles, label
from game.star import (
    WIN_PERCENT,
    as_puzzle,
    check_star_word,
    get_star_puzzle,
    rank_for,
    star_hint,
)
from game.wordle_scoring import evaluate_wordle_result

st.set_page_config(page_title="AI Word Arena", page_icon="🎯")

st.title("AI Word Arena")


# -----------------------------
# Session state
# -----------------------------
def init_state():
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = DifficultyManager()

    defaults = {
        "used_words": [],
        "guesses": [],
        "wordle_status": None,
        "round_done": False,
        "last_result": None,
        "hints": [],
        "show_summary": False,
        "scramble": "",
        "anagram_attempts": 0,
        "anagram_status": None,
        "star": None,
        "star_found": [],
        "star_points": 0,
        "star_outer": [],
        "star_flash": None,
        "theme_notice": "",
        "agent_trace": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# -----------------------------
# Saving progress (logged-in players only; guests are never saved)
# -----------------------------
def can_save():
    return bool(st.session_state.get("persist"))


def save_progress():
    if not can_save():
        return

    player = st.session_state.player

    try:
        store.save_progress(
            player.name,
            player,
            st.session_state.difficulty,
            st.session_state.used_words,
        )

        st.session_state.save_error = ""
    except Exception as exc:  # a database problem must never stop the game
        st.session_state.save_error = f"{type(exc).__name__}: {exc}"


def persist_round(puzzle, correct, points):
    if not can_save():
        return

    try:
        store.log_round(
            st.session_state.player.name,
            st.session_state.game_mode,
            puzzle.theme,
            puzzle.difficulty,
            puzzle.word,
            correct,
            points,
            hints=len(st.session_state.hints),
            source=puzzle.source,
        )
    except Exception as exc:
        st.session_state.save_error = f"{type(exc).__name__}: {exc}"
        return

    save_progress()


# -----------------------------
# AI allowance (public mode limits how much AI one visitor can use)
# -----------------------------
def ai_calls_left():
    """None = unlimited; otherwise how many AI hints/feedback this visit may still use."""
    if not config.PUBLIC_MODE or not config.AI_SESSION_LIMIT:
        return None

    return max(0, config.AI_SESSION_LIMIT - st.session_state.get("ai_calls", 0))


def ai_allowed():
    if not llm_client.is_available():
        return False

    left = ai_calls_left()

    return left is None or left > 0


def note_ai_use():
    st.session_state.ai_calls = st.session_state.get("ai_calls", 0) + 1


# -----------------------------
# Round lifecycle
# -----------------------------
def current_difficulty_name():
    level = st.session_state.difficulty.get_level()

    return DIFFICULTY_LEVELS[level - 1]


def start_round(theme, mode):
    st.session_state.theme = theme
    st.session_state.game_mode = mode

    st.session_state.guesses = []
    st.session_state.wordle_status = None
    st.session_state.round_done = False
    st.session_state.last_result = None
    st.session_state.hints = []
    st.session_state.scramble = ""
    st.session_state.anagram_attempts = 0
    st.session_state.anagram_status = None
    st.session_state.star = None
    st.session_state.star_found = []
    st.session_state.star_points = 0
    st.session_state.star_outer = []
    st.session_state.star_flash = None
    st.session_state.theme_notice = ""
    st.session_state.pop("correct_word", None)

    use_agent = (
        bool(st.session_state.get("use_agent"))
        and llm_client.is_available()
        and not config.PUBLIC_MODE
    )

    agent_log.clear_last()
    st.session_state.agent_trace = None

    with st.spinner("Getting your puzzle..." if not use_agent else "The agent is building and checking your puzzle..."):
        if mode == "Star":
            star = get_star_puzzle(
                theme,
                current_difficulty_name(),
                st.session_state.used_words,
            )

            st.session_state.star = star

            if star is not None:
                st.session_state.star_outer = list(star.outer)

            puzzle = (
                as_puzzle(star, current_difficulty_name())
                if star is not None
                else None
            )
        else:
            puzzle = get_puzzle(
                theme,
                current_difficulty_name(),
                mode,
                st.session_state.used_words,
                use_agent=use_agent,
            )

            st.session_state.agent_trace = agent_log.last_run()

    st.session_state.puzzle = puzzle

    if (
        puzzle is not None
        and mode != "Star"
        and puzzle.theme.casefold() != theme.casefold()
    ):
        st.session_state.theme_notice = (
            f"No puzzles for '{theme}' are available right now (the AI is "
            f"unavailable), so here is a {puzzle.theme} puzzle instead."
        )

    if puzzle is not None:
        st.session_state.used_words.append(puzzle.word)

        if mode == "Anagram":
            st.session_state.scramble = scramble(puzzle.word)


def finish_round(correct, performance=None, bonus=0):
    """
    Called exactly once per round. Updates score, streak, per-theme stats
    and the adaptive difficulty, then saves. Returns the points earned.
    `bonus` is extra points on top (Star word points).
    """
    player = st.session_state.player
    puzzle = st.session_state.puzzle

    st.session_state.difficulty.record_result(
        correct if performance is None else performance
    )

    points = update_player_score(player, correct)

    player.total_score += bonus

    record_round(player, puzzle.theme, correct)

    st.session_state.round_done = True

    persist_round(puzzle, correct, points + bonus)

    return points


# -----------------------------
# Account settings (change PIN, delete my data)
# -----------------------------
def render_account_settings():
    name = st.session_state.player.name

    with st.expander("🔒 Account"):
        st.write("**Change PIN**")

        with st.form("change_pin_form", clear_on_submit=True):
            old = st.text_input("Current PIN", type="password", max_chars=8, key="cp_old")
            new = st.text_input("New PIN", type="password", max_chars=8, key="cp_new")
            new2 = st.text_input("New PIN again", type="password", max_chars=8, key="cp_new2")

            changed = st.form_submit_button("Change PIN")

        if changed:
            problem = pin_problem(new)

            if not clean_pin(old):
                st.warning("Type your current PIN (4 to 8 digits).")
            elif problem:
                st.warning(problem)
            elif new != new2:
                st.warning("The two new PINs do not match.")
            else:
                try:
                    store.change_pin(name, old, new)
                    st.success("Your PIN was changed.")
                except store.WrongPin as err:
                    st.warning(f"That is not your current PIN ({err.attempts_left} tries left).")
                except store.Locked as err:
                    st.warning(lock_message(err.seconds))
                except Exception:
                    st.warning("Could not change the PIN right now.")

        st.write("**Delete my saved data**")
        st.caption(
            "Removes your profile and round history for good. "
            "This cannot be undone."
        )

        with st.form("delete_form", clear_on_submit=True):
            pin = st.text_input("Your PIN", type="password", max_chars=8, key="del_pin")
            sure = st.checkbox("Yes, delete everything saved about me", key="del_sure")

            deleted = st.form_submit_button("Delete my saved data")

        if deleted:
            if not sure:
                st.warning("Tick the box to confirm.")
            elif not clean_pin(pin):
                st.warning("Type your PIN (4 to 8 digits).")
            else:
                try:
                    store.delete_player(name, pin)
                except store.WrongPin as err:
                    st.warning(f"Wrong PIN ({err.attempts_left} tries left).")
                except store.Locked as err:
                    st.warning(lock_message(err.seconds))
                except Exception:
                    st.warning("Could not delete right now.")
                else:
                    st.session_state.clear()
                    st.session_state.flash = "Your saved data was deleted."
                    st.rerun()


# -----------------------------
# Sidebar: live stats
# -----------------------------
def render_sidebar():
    player = st.session_state.player

    with st.sidebar:
        st.header(f"{player.name}'s progress")

        st.metric("Score", player.total_score)
        st.metric("Streak", player.current_streak)
        st.metric("Difficulty", current_difficulty_name())

        left = ai_calls_left()

        if llm_client.is_available() and left == 0:
            st.caption("AI hints and feedback: this session's allowance is used up - using built-in ones")
        elif llm_client.is_available():
            st.caption(
                "AI hints and feedback: on"
                + (f" ({left} left this session)" if left is not None else "")
            )

        if llm_client.is_available() and not config.PUBLIC_MODE:
            st.checkbox(
                "🧠 Use the AI agent (slower, checks its own work)",
                value=False,
                key="use_agent",
                help=(
                    "The agent checks each word and sentence with dictionary tools "
                    "and has a second model review the puzzle. On a free plan that "
                    "takes about a minute per puzzle, so it is best used to build the "
                    "puzzle bank ahead of time (scripts/build_puzzle_bank.py --agent)."
                ),
            )
        else:
            st.caption("AI is off - using built-in hints and feedback")

        if not can_save():
            st.caption("Guest progress is not saved.")
        elif st.session_state.get("save_error"):
            st.warning("⚠️ Your progress could not be saved.")
        else:
            st.caption("✅ Progress is saved automatically.")

        if st.button("Log out", key="log_out"):
            st.session_state.clear()
            st.rerun()

        if can_save():
            render_account_settings()

        runs = agent_log.load_runs(limit=200)

        if runs:
            summary = agent_log.summarize_runs(runs)

            with st.expander("📈 Agent quality"):
                st.metric("Agent runs", summary["runs"])
                st.metric("Success rate", f"{summary['success_rate']}%")
                st.caption(
                    f"Average {summary['avg_steps']} steps, "
                    f"{summary['avg_llm_calls']} AI calls, "
                    f"{summary['avg_seconds']}s per puzzle"
                    + (f", about {summary['avg_tokens']} tokens" if summary["avg_tokens"] else "")
                )

                if summary["top_rejections"]:
                    st.write("**Most common rejections**")

                    for reason, count in summary["top_rejections"]:
                        st.write(f"- {reason} ({count})")


# -----------------------------
# Hints
# -----------------------------
def render_hints(puzzle):
    hints = st.session_state.hints

    out_of_hints = len(hints) >= MAX_HINTS

    # The label must stay constant (Streamlit ties a button's identity to its
    # label), so the counter is shown separately.
    clicked = st.button(
        "💡 Need another hint?",
        key="hint_button",
        disabled=st.session_state.round_done or out_of_hints,
    )

    st.caption(f"Hints used: {len(hints)}/{MAX_HINTS}")

    if clicked and not out_of_hints and not st.session_state.round_done:
        if st.session_state.game_mode == "Star":
            text = star_hint(
                st.session_state.star,
                st.session_state.star_found,
                len(hints),
            )
        else:
            with st.spinner("Thinking of a hint..."):
                text, _from_ai = get_extra_hint(
                    puzzle, hints, use_ai=ai_allowed(), on_call=note_ai_use
                )

        hints.append(text)
        st.session_state.player.hints_used += 1
        save_progress()

        st.rerun()  # refresh so the counter and button state are current

    for i, hint in enumerate(hints, start=1):
        st.info(f"Hint {i}: {hint}")


# -----------------------------
# Agent trace (only after the round: it contains the answer)
# -----------------------------
def render_agent_trace():
    trace = st.session_state.agent_trace

    if not trace:
        return

    title = (
        "🧠 How the agent built this puzzle"
        if trace["success"]
        else "🧠 The agent could not finish this one (another source was used)"
    )

    with st.expander(title):
        st.caption(
            f"{len(trace['steps'])} steps · {trace['llm_calls']} AI calls · "
            f"{trace['seconds']}s · {trace['reason']}"
        )

        for step in trace["steps"]:
            icon = "✅" if step["ok"] else "⚠️"
            thought = f" - {step['thought']}" if step["thought"] else ""

            st.markdown(f"{icon} **{step['i']}. `{step['action']}`**{thought}")
            st.caption(step["observation"])


# -----------------------------
# Result display
# -----------------------------
def render_result(puzzle):
    if st.session_state.last_result:
        kind, text = st.session_state.last_result

        if kind == "success":
            st.success(text)
        elif kind == "info":
            st.info(text)
        else:
            st.error(text)

    if st.session_state.round_done:
        if puzzle.explanation:
            st.info(f"💬 {puzzle.explanation}")

        render_agent_trace()

        if st.button("Next puzzle ➡️", key="next_puzzle"):
            start_round(
                st.session_state.theme,
                st.session_state.game_mode,
            )

            st.rerun()


# -----------------------------
# MCQ mode
# -----------------------------
def render_mcq(puzzle):
    answer = st.radio(
        "Choose your answer:",
        puzzle.options,
        index=None,  # nothing pre-selected: the player must actually choose
        key=f"mcq_{puzzle.word}_{len(st.session_state.used_words)}",
        disabled=st.session_state.round_done,
    )

    submitted = st.button(
        "Submit Answer",
        key="mcq_submit",
        disabled=st.session_state.round_done or answer is None,
    )

    if submitted and answer is not None and not st.session_state.round_done:
        result = check_answer(puzzle, answer)

        points = finish_round(result.correct)

        if result.correct:
            st.session_state.last_result = (
                "success",
                f"{result.message} +{points} points",
            )
        else:
            st.session_state.last_result = ("error", result.message)

        st.rerun()  # refresh so the button locks and the result shows


# -----------------------------
# Wordle mode
# -----------------------------
BOX_STYLE = (
    "display:inline-flex;justify-content:center;align-items:center;"
    "width:50px;height:50px;font-size:28px;font-weight:bold;margin:3px"
)

COLORS = {"green": "#6aaa64", "yellow": "#c9b458", "gray": "#787c7e"}


def wordle_row_html(word="", feedback=None):
    cells = []

    for i in range(WORD_LENGTH):
        if i < len(word):
            bg = COLORS.get(feedback[i], COLORS["gray"])

            cells.append(
                f'<div style="{BOX_STYLE};background-color:{bg};color:white;'
                f'border-radius:3px;">{word[i]}</div>'
            )
        else:
            cells.append(
                f'<div style="{BOX_STYLE};border:2px solid #d3d6da;"></div>'
            )

    return "<div>" + "".join(cells) + "</div>"


def handle_guess(puzzle, raw_guess):
    """Processes one guess. Returns a warning string, or None if accepted."""
    guess = raw_guess.upper().strip()

    if len(guess) != WORD_LENGTH:
        return f"Please enter exactly {WORD_LENGTH} letters."

    if not guess.isalpha():
        return "Only letters are allowed."

    previous = [g["word"] for g in st.session_state.guesses]

    if guess in previous:
        return f"You already tried {guess}. Please enter a different word."

    result = check_wordle_answer(puzzle, guess)

    if not result["valid"]:
        return result["message"]

    st.session_state.guesses.append(
        {"word": result["word"], "feedback": result["feedback"]}
    )

    used = len(st.session_state.guesses)

    if result["correct"]:
        st.session_state.wordle_status = "won"

        points = finish_round(True, evaluate_wordle_result(used, True))

        st.session_state.last_result = (
            "success",
            f"Correct! 🎉 +{points} points",
        )

    elif used >= MAX_WORDLE_ATTEMPTS:
        st.session_state.wordle_status = "lost"

        finish_round(False, evaluate_wordle_result(used, False))

        st.session_state.last_result = (
            "error",
            f"Game over! The answer was {result['answer']}.",
        )

    return None


def render_wordle(puzzle):
    guesses = st.session_state.guesses

    rows = []

    for i in range(MAX_WORDLE_ATTEMPTS):
        if i < len(guesses):
            rows.append(wordle_row_html(guesses[i]["word"], guesses[i]["feedback"]))
        else:
            rows.append(wordle_row_html())

    st.markdown("".join(rows), unsafe_allow_html=True)

    if st.session_state.wordle_status is None:
        with st.form("wordle_form", clear_on_submit=True):
            guess = st.text_input(f"Enter your {WORD_LENGTH}-letter word:")

            submitted = st.form_submit_button("Submit Guess")

        if submitted:
            warning = handle_guess(puzzle, guess)

            if warning:
                st.warning(warning)
            else:
                st.rerun()


# -----------------------------
# Anagram mode
# -----------------------------
def anagram_tiles_html(letters):
    cells = "".join(
        f'<div style="{BOX_STYLE};background-color:#3b6ea5;color:white;'
        f'border-radius:6px;">{c}</div>'
        for c in letters
    )

    return f"<div>{cells}</div>"


def handle_anagram_guess(puzzle, raw_guess):
    """Returns a warning string, or None if the round state changed."""
    result = check_anagram(puzzle, st.session_state.scramble, raw_guess)

    if not result["counts"]:
        return result["message"]  # malformed input costs no attempt

    st.session_state.anagram_attempts += 1

    used = st.session_state.anagram_attempts

    if result["correct"]:
        st.session_state.anagram_status = "won"

        points = finish_round(True, used <= 2)

        st.session_state.last_result = (
            "success",
            f"{result['message']} +{points} points",
        )

        return None

    if used >= MAX_ANAGRAM_ATTEMPTS:
        end_anagram_round(puzzle)

        return None

    left = MAX_ANAGRAM_ATTEMPTS - used

    return f"{result['message']} {left} attempt(s) left."


def end_anagram_round(puzzle):
    st.session_state.anagram_status = "lost"

    finish_round(False)

    st.session_state.last_result = (
        "error",
        f"The answer was {puzzle.word}.",
    )


def render_anagram(puzzle):
    st.markdown(
        anagram_tiles_html(st.session_state.scramble),
        unsafe_allow_html=True,
    )

    st.caption(
        f"Attempts: {st.session_state.anagram_attempts}/{MAX_ANAGRAM_ATTEMPTS}"
    )

    if st.session_state.anagram_status is None:
        with st.form("anagram_form", clear_on_submit=True):
            guess = st.text_input("Unscramble the letters:")

            submitted = st.form_submit_button("Submit Guess")

        if submitted:
            warning = handle_anagram_guess(puzzle, guess)

            if warning:
                st.warning(warning)
            else:
                st.rerun()

        if st.button("Give up", key="anagram_give_up"):
            end_anagram_round(puzzle)

            st.rerun()


# -----------------------------
# Star mode (Spelling-Bee style)
# -----------------------------
TILE = (
    "display:inline-flex;justify-content:center;align-items:center;"
    "width:58px;height:58px;font-size:26px;font-weight:bold;margin:4px;"
    "border-radius:10px"
)


def star_layout_html(center, outer):
    def tile(letter, is_center=False):
        style = (
            f"{TILE};background-color:#f7da21;color:#222;"
            if is_center
            else f"{TILE};background-color:#e6e6e6;color:#222;"
        )

        return f'<div style="{style}">{letter}</div>'

    top = tile(outer[0]) + tile(outer[1])
    middle = tile(outer[2]) + tile(center, True) + tile(outer[3])
    bottom = tile(outer[4]) + tile(outer[5])

    return (
        '<div style="text-align:center">'
        f"<div>{top}</div><div>{middle}</div><div>{bottom}</div>"
        "</div>"
    )


def finish_star_round(puzzle):
    star = st.session_state.star
    found = st.session_state.star_found
    points = st.session_state.star_points

    percent = round(100 * points / star.max_points) if star.max_points else 0
    won = percent >= WIN_PERCENT

    # Word points count towards the total score, on top of the round result
    finish_round(won, won, bonus=points)

    st.session_state.last_result = (
        "success" if won else "info",
        f"Round over: {len(found)}/{len(star.words)} words, "
        f"{points}/{star.max_points} points - rank "
        f"{rank_for(points, star.max_points)}. "
        f"Pangram: {', '.join(star.pangrams)}.",
    )


def handle_star_word(raw):
    star = st.session_state.star

    result = check_star_word(star, st.session_state.star_found, raw)

    if not result["ok"]:
        return result["message"]

    word = raw.strip().upper()

    st.session_state.star_found.append(word)
    st.session_state.star_points += result["points"]

    prefix = "⭐ Pangram! " if result["pangram"] else "Nice! "

    st.session_state.star_flash = ("success", f"{prefix}{word} {result['message']}")

    return None


def render_star(puzzle):
    star = st.session_state.star

    if st.session_state.star_flash:
        st.success(st.session_state.star_flash[1])
        st.session_state.star_flash = None

    st.markdown(
        star_layout_html(star.center, st.session_state.star_outer or star.outer),
        unsafe_allow_html=True,
    )

    found = st.session_state.star_found
    points = st.session_state.star_points

    st.progress(min(1.0, points / star.max_points) if star.max_points else 0.0)

    st.caption(
        f"Rank: **{rank_for(points, star.max_points)}** · "
        f"{points}/{star.max_points} points · "
        f"{len(found)}/{len(star.words)} words"
    )

    if not st.session_state.round_done:
        with st.form("star_form", clear_on_submit=True):
            word = st.text_input(
                f"Type a word (4+ letters, must use {star.center}):"
            )

            submitted = st.form_submit_button("Submit Word")

        if submitted:
            warning = handle_star_word(word)

            if warning:
                st.warning(warning)
            else:
                st.rerun()

        left, right = st.columns(2)

        if left.button("🔀 Shuffle letters", key="star_shuffle"):
            letters = list(st.session_state.star_outer or star.outer)

            random.shuffle(letters)

            st.session_state.star_outer = letters

            st.rerun()

        if right.button("Finish round", key="star_finish"):
            finish_star_round(puzzle)

            st.rerun()

    if found:
        shown = [f"⭐ {w}" if set(w) == set(star.letters) else w for w in sorted(found)]

        st.write("**Found:** " + ", ".join(shown))

    if st.session_state.round_done:
        missed = [w for w in star.words if w not in found]

        if missed:
            with st.expander(f"Words you missed ({len(missed)})"):
                st.write(", ".join(missed))


# -----------------------------
# Puzzle screen
# -----------------------------
def render_puzzle(puzzle):
    st.subheader(f"{puzzle.theme} · {puzzle.difficulty}")

    source_labels = {
        "ai": "🤖 Generated live by AI",
        "bank": "📚 From the puzzle bank",
        "local": "📦 Built-in puzzle",
        "lexicon": "📖 Built from the dictionary",
        "agent": "🧠 Built by the AI agent and checked with dictionary tools",
    }

    if puzzle.source in source_labels:
        st.caption(source_labels[puzzle.source])

    if st.session_state.theme_notice:
        st.info(st.session_state.theme_notice)

    mode = st.session_state.game_mode

    if mode == "MCQ":
        st.write(puzzle.sentence)
    elif mode == "Star":
        st.write(
            "Make words from the letters. Each word needs **4+ letters** and "
            "must use the **yellow center letter**. One word uses all 7 letters!"
        )
    else:
        st.write(f"**Clue:** {puzzle.hint or puzzle.sentence}")

    render_hints(puzzle)

    if mode == "MCQ":
        render_mcq(puzzle)
    elif mode == "Wordle":
        render_wordle(puzzle)
    elif mode == "Anagram":
        render_anagram(puzzle)
    else:
        render_star(puzzle)

    render_result(puzzle)


# -----------------------------
# End-of-session summary
# -----------------------------
def render_summary():
    player = st.session_state.player

    summary = build_summary(
        player,
        st.session_state.difficulty.get_level(),
    )

    st.subheader("📊 Your progress summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rounds", summary["games_played"])
    c2.metric("Accuracy", f"{summary['accuracy']}%")
    c3.metric("Best streak", summary["best_streak"])
    c4.metric("Hints used", summary["hints_used"])

    if summary["themes"]:
        st.table(
            [
                {
                    "Theme": t["theme"],
                    "Played": t["played"],
                    "Correct": t["correct"],
                    "Accuracy": f"{t['accuracy']}%",
                }
                for t in summary["themes"]
            ]
        )

    if can_save():
        try:
            recent = store.recent_rounds(player.name, 10)
        except Exception:
            recent = []

        if recent:
            st.write("**Your recent rounds**")
            st.table(
                [
                    {
                        "Mode": r["mode"],
                        "Category": r["theme"],
                        "Level": r["difficulty"],
                        "Result": "✅" if r["correct"] else "❌",
                        "Points": r["points"],
                    }
                    for r in recent
                ]
            )

    # Cache so the AI is only called when the stats actually change
    cache_key = (summary["games_played"], summary["hints_used"])

    cached = st.session_state.get("feedback_cache")

    if cached is None or cached[0] != cache_key:
        with st.spinner("Writing your feedback..."):
            text, from_ai = get_session_feedback(
                summary, use_ai=ai_allowed(), on_call=note_ai_use
            )

        st.session_state.feedback_cache = (cache_key, text, from_ai)

    _key, text, from_ai = st.session_state.feedback_cache

    st.write(text)

    st.caption(
        "Feedback written by AI"
        if from_ai
        else "Auto-generated feedback (AI is not connected)"
    )


# -----------------------------
# Welcome screen: ask who is playing
# -----------------------------
def start_session(loaded, note):
    st.session_state.player = loaded.player
    st.session_state.difficulty = loaded.difficulty
    st.session_state.used_words = loaded.used_words
    st.session_state.persist = True
    st.session_state.welcome_note = note

    st.rerun()


def start_unsaved(name):
    """The database is unavailable: still let the person play, and say so."""
    st.session_state.player = Player(name=name)
    st.session_state.persist = False
    st.session_state.welcome_note = (
        f"Hi {name}! Saving is unavailable right now, so this session "
        "will not be kept."
    )

    st.rerun()


def lock_message(seconds):
    minutes = max(1, math.ceil(seconds / 60))

    return (
        "Too many wrong PINs, so this profile is locked for a short while. "
        f"Try again in about {minutes} minute{'s' if minutes != 1 else ''}."
    )


def log_in(name, pin):
    try:
        loaded = store.authenticate(name, pin)
    except store.NoSuchPlayer:
        st.warning(
            f"There is no player called {name}. Check the spelling, or create "
            "a profile in the New player tab."
        )
        return
    except store.WrongPin as err:
        left = err.attempts_left

        st.warning(
            f"Wrong PIN. {left} more wrong {'try' if left == 1 else 'tries'} "
            "before this profile is locked for a few minutes."
        )
        return
    except store.Locked as err:
        st.warning(lock_message(err.seconds))
        return
    except Exception as exc:  # e.g. the database file is locked or unwritable
        st.session_state.save_error = f"{type(exc).__name__}: {exc}"
        start_unsaved(name)
        return

    p = loaded.player

    if loaded.pin_was_set:
        note = (
            f"Welcome back, {p.name}! This profile was created before PINs "
            "existed, so the PIN you just typed is now its PIN. Remember it!"
        )
    else:
        note = (
            f"Welcome back, {p.name}! You have {p.total_score} points "
            f"from {p.games_played} rounds."
        )

    start_session(loaded, note)


def sign_up(name, pin):
    try:
        loaded = store.create_player(name, pin)
    except store.NameTaken:
        st.warning(
            f"The name {name} is already taken. If it is yours, use the Log in "
            "tab; otherwise choose a different name."
        )
        return
    except Exception as exc:
        st.session_state.save_error = f"{type(exc).__name__}: {exc}"
        start_unsaved(name)
        return

    start_session(
        loaded,
        f"Nice to meet you, {loaded.player.name}! Your profile is created "
        "and your progress will be saved. Remember your PIN.",
    )


def play_as_guest():
    st.session_state.player = Player(name=GUEST_NAME)
    st.session_state.persist = False
    st.session_state.welcome_note = "Playing as a guest - progress will not be saved."

    st.rerun()


def render_leaderboard():
    try:
        rows = store.leaderboard(10)
    except Exception:
        return

    if not rows:
        return

    with st.expander("🏆 Top players"):
        st.table(
            [
                {
                    "Player": r["name"],
                    "Score": r["score"],
                    "Rounds": r["rounds"],
                    "Accuracy": f"{r['accuracy']}%",
                }
                for r in rows
            ]
        )


def render_login_tab():
    with st.form("login_form", clear_on_submit=True):
        typed = st.text_input("Your name", max_chars=20, key="login_name")
        pin = st.text_input("Your PIN", type="password", max_chars=8, key="login_pin")

        submitted = st.form_submit_button("Log in")

    if not submitted:
        return

    name = clean_name(typed)

    if not name:
        st.warning("Please type your name (letters or numbers).")
    elif not clean_pin(pin):
        st.warning("Your PIN is 4 to 8 digits.")
    else:
        log_in(name, pin)


def render_signup_tab():
    st.caption(
        "Pick a name and a 4 to 8 digit PIN. You will need both to come back. "
        "There is no way to recover a forgotten PIN yourself."
    )

    with st.form("signup_form", clear_on_submit=True):
        typed = st.text_input("Your name", max_chars=20, key="new_name")
        pin = st.text_input("Choose a PIN", type="password", max_chars=8, key="new_pin")
        again = st.text_input("Type the PIN again", type="password", max_chars=8, key="new_pin2")

        submitted = st.form_submit_button("Create profile")

    if not submitted:
        return

    name = clean_name(typed)
    problem = pin_problem(pin)

    if not name:
        st.warning("Please type a name (letters or numbers).")
    elif problem:
        st.warning(problem)
    elif pin != again:
        st.warning("The two PINs do not match.")
    else:
        sign_up(name, pin)


def render_name_screen():
    st.subheader("Welcome! 👋")

    if config.PUBLIC_MODE:
        st.info(
            "This is a public demo. Saved profiles can be reset whenever the app "
            "restarts, and names on the Top players list are visible to everyone - "
            "so please do not use your real name, and do not reuse a PIN from anywhere else."
        )

    if st.session_state.get("flash"):
        st.info(st.session_state.flash)
        st.session_state.flash = ""

    login_tab, signup_tab = st.tabs(["Log in", "New player"])

    with login_tab:
        render_login_tab()

    with signup_tab:
        render_signup_tab()

    if st.button("Play as guest (progress not saved)", key="play_as_guest"):
        play_as_guest()

    render_leaderboard()


if "player" not in st.session_state:
    render_name_screen()
    st.stop()


# -----------------------------
# Main layout
# -----------------------------
render_sidebar()

st.caption(f"Playing as **{st.session_state.player.name}**")

if st.session_state.get("welcome_note"):
    st.success(st.session_state.welcome_note)
    st.session_state.welcome_note = ""

choice = st.selectbox(
    "Choose your category",
    SECTORS + ([] if config.PUBLIC_MODE else [CUSTOM_LABEL]),
    format_func=lambda c: c if c == CUSTOM_LABEL else label(c),
)

can_start = True

if choice == CUSTOM_LABEL:
    typed = st.text_input(
        "Type any topic (for example Formula 1, Bollywood, Chess):",
        max_chars=40,
        key="custom_topic",
    )

    theme = clean_topic(typed)

    if not theme:
        can_start = False
    elif not (llm_client.is_available() or has_offline_puzzles(theme)):
        can_start = False

        st.info(
            "Custom topics need the AI to be connected (set your API key). "
            "Pick a category above to play without AI."
        )
else:
    theme = choice

game_mode = st.selectbox(
    "Choose your game mode", ["MCQ", "Wordle", "Anagram", "Star"]
)

if st.button("Start Game", key="start_game", disabled=not can_start):
    start_round(theme, game_mode)

if "puzzle" in st.session_state:
    if st.session_state.puzzle is None:
        st.error("Could not find a puzzle. Please try again.")
    else:
        render_puzzle(st.session_state.puzzle)

st.divider()

if st.button("📊 End session & get feedback", key="end_session"):
    st.session_state.show_summary = True

if st.session_state.show_summary:
    render_summary()
