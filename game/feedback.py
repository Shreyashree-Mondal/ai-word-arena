"""Session statistics and end-of-session feedback (AI-written, with fallback)."""

import config
from ai import llm_client
from game.players import is_named


def record_round(player, theme, correct):
    """Track per-theme results for the feedback summary."""
    stats = player.theme_stats.setdefault(theme, {"played": 0, "correct": 0})

    stats["played"] += 1

    if correct:
        stats["correct"] += 1


def build_summary(player, difficulty_level):
    played = player.games_played
    correct = player.correct_answers

    accuracy = round(100 * correct / played) if played else 0

    themes = []

    for name, s in player.theme_stats.items():
        themes.append(
            {
                "theme": name,
                "played": s["played"],
                "correct": s["correct"],
                "accuracy": round(100 * s["correct"] / s["played"]),
            }
        )

    themes.sort(key=lambda t: (-t["played"], t["theme"]))

    strongest = weakest = None

    if themes:
        ranked = sorted(themes, key=lambda t: (-t["accuracy"], -t["played"]))
        strongest = ranked[0]

        worst = ranked[-1]

        # Only call something a weak spot if it is actually weaker
        if worst["accuracy"] < strongest["accuracy"]:
            weakest = worst

    return {
        "name": player.name,
        "games_played": played,
        "correct": correct,
        "accuracy": accuracy,
        "best_streak": player.best_streak,
        "hints_used": player.hints_used,
        "score": player.total_score,
        "difficulty": config.DIFFICULTY_LEVELS[difficulty_level - 1],
        "themes": themes,
        "strongest": strongest,
        "weakest": weakest,
    }


def rule_based_feedback(summary):
    if summary["games_played"] == 0:
        return "Play a few rounds first, then come back for your feedback."

    acc = summary["accuracy"]

    who = f", {summary['name']}" if is_named(summary.get("name")) else ""

    if acc >= 80:
        opening = f"Excellent work{who}!"
    elif acc >= 50:
        opening = f"Solid progress{who}!"
    else:
        opening = f"Good effort{who} - this is how vocabulary grows."

    parts = [
        f"{opening} You got {summary['correct']} of "
        f"{summary['games_played']} rounds right ({acc}%), "
        f"with a best streak of {summary['best_streak']}."
    ]

    if summary["strongest"]:
        s = summary["strongest"]
        parts.append(f"Your strongest theme was {s['theme']} ({s['accuracy']}%).")

    if summary["weakest"]:
        w = summary["weakest"]
        parts.append(
            f"{w['theme']} needs more practice ({w['accuracy']}%) - "
            "try a few more rounds there."
        )

    if summary["hints_used"]:
        parts.append(
            f"You used {summary['hints_used']} hint(s); "
            "try guessing a little longer before asking next time."
        )

    parts.append(f"You are currently playing at {summary['difficulty']} difficulty.")

    return " ".join(parts)


def _build_prompt(summary):
    theme_lines = "\n".join(
        f"- {t['theme']}: {t['correct']}/{t['played']} correct ({t['accuracy']}%)"
        for t in summary["themes"]
    ) or "- none"

    return f"""A player just finished a vocabulary word game. Write short, friendly feedback.

Facts (use ONLY these, do not invent anything):
- Rounds played: {summary['games_played']}
- Correct: {summary['correct']} ({summary['accuracy']}%)
- Best streak: {summary['best_streak']}
- Hints used: {summary['hints_used']}
- Current difficulty: {summary['difficulty']}
- Results by theme:
{theme_lines}

Write 3 to 4 sentences: praise something specific, name one theme to practice
(if the results show a weaker one), and give one concrete tip.
Return only the feedback text."""


def get_session_feedback(summary, use_ai=True, on_call=None):
    """Returns (text, from_ai). Falls back to rule-based text."""
    if use_ai and summary["games_played"] > 0 and llm_client.is_available():
        if on_call:
            on_call()   # counts the request itself, whether or not the reply turns out usable

        text = llm_client.generate_text(
            _build_prompt(summary),
            max_tokens=220,
            temperature=0.6,
        )

        if text and 20 <= len(text.strip()) <= 900:
            return text.strip(), True

    return rule_based_feedback(summary), False
