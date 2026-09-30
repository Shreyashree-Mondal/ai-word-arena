"""Connects the agent to the game: run, log, optionally save to the bank."""

import config
from agent import log
from agent.puzzle_agent import PuzzleAgent
from ai.puzzle_bank import append_entry, entry_to_puzzle, load_entries


def _theme_words(entries, theme):
    """Words already in the bank for this theme (shown to the model so it avoids them)."""
    wanted = (theme or "").strip().casefold()

    return [
        e.get("word", "")
        for e in entries
        if str(e.get("theme", "")).strip().casefold() == wanted and e.get("word")
    ]


def generate_with_agent(
    theme,
    difficulty,
    used_words=None,
    llm=None,
    bank_path=None,
    log_path=None,
    clock=None,
):
    """A validated Puzzle built by the agent, or None (the caller then falls back)."""
    entries = load_entries(bank_path)

    bank_words = [e.get("word", "") for e in entries]
    theme_words = _theme_words(entries, theme)

    kwargs = {"clock": clock} if clock else {}

    agent = PuzzleAgent(llm=llm, **kwargs)

    result = agent.run(theme, difficulty, used_words or [], bank_words, theme_words)

    log.record_run(result, log_path)

    if not result.success:
        return None

    if config.AUTO_SAVE_AI_PUZZLES:
        try:
            append_entry(result.entry, bank_path)
        except OSError as exc:
            print(f"[agent] could not save puzzle to the bank: {exc}")

    puzzle = entry_to_puzzle(result.entry)
    puzzle.source = "agent"

    return puzzle


def build_bank_with_agent(
    themes,
    difficulties,
    target,
    max_runs_per_combo=10,
    llm=None,
    bank_path=None,
    log_path=None,
    log_fn=print,
):
    """Top up the bank to `target` puzzles per theme+difficulty using the agent."""
    stats = {"added": 0, "runs": 0, "failed_runs": 0, "aborted": False}

    for theme in themes:
        for difficulty in difficulties:

            def have():
                return sum(
                    1
                    for e in load_entries(bank_path)
                    if e.get("theme") == theme and e.get("difficulty") == difficulty
                )

            runs = 0

            while have() < target and runs < max_runs_per_combo:
                runs += 1
                stats["runs"] += 1

                before = have()

                entries = load_entries(bank_path)

                bank_words = [e.get("word", "") for e in entries]

                result = PuzzleAgent(llm=llm).run(
                    theme, difficulty, [], bank_words, _theme_words(entries, theme)
                )

                log.record_run(result, log_path)

                if result.reason == "LLM unavailable":
                    stats["aborted"] = True
                    log_fn("AI unavailable - stopping. Progress is saved.")
                    return stats

                if result.success and append_entry(result.entry, bank_path):
                    stats["added"] += 1
                elif not result.success:
                    stats["failed_runs"] += 1

                assert have() >= before

            log_fn(f"{theme} / {difficulty}: {have()}/{target}")

    return stats
