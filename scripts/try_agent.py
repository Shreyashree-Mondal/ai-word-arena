"""
Watch the AI agent work, step by step, in the terminal.

    export GROQ_API_KEY="your-key"
    python scripts/try_agent.py "Formula 1" Easy
    python scripts/try_agent.py Cricket Hard --no-critic
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from agent.puzzle_agent import PuzzleAgent  # noqa: E402
from ai import llm_client  # noqa: E402
from ai.puzzle_bank import load_entries  # noqa: E402


def main(argv=None, llm=None, out=print):
    parser = argparse.ArgumentParser()
    parser.add_argument("theme", nargs="?", default="Space")
    parser.add_argument("difficulty", nargs="?", default="Easy",
                        choices=["Easy", "Medium", "Hard"])
    parser.add_argument("--no-critic", action="store_true")
    args = parser.parse_args(argv)

    if llm is None and not llm_client.is_available():
        out("AI is not configured. Set your key first, e.g.:")
        out('  export GROQ_API_KEY="your-key"')
        return 1

    # This script can afford to wait, so wait out rate limits instead of failing
    config.API_PACING_MAX_WAIT = max(config.API_PACING_MAX_WAIT, 90)
    config.API_RETRY_MAX_WAIT = max(config.API_RETRY_MAX_WAIT, 60)

    bank_words = [e.get("word", "") for e in load_entries()]

    agent = PuzzleAgent(llm=llm, use_critic=not args.no_critic)

    out(f"Agent starting: theme={args.theme!r}, difficulty={args.difficulty}")

    result = agent.run(args.theme, args.difficulty, [], bank_words)

    for step in result.steps:
        mark = "OK  " if step.ok else "FAIL"

        out(f"\n[{step.index}] {mark} {step.action}  {json.dumps(step.args, ensure_ascii=False)[:200]}")

        if step.thought:
            out(f"     thought: {step.thought}")

        out(f"     result:  {json.dumps(step.observation, ensure_ascii=False)[:300]}")

    tokens = f", {result.tokens} tokens" if result.tokens else ""

    out(f"\n{result.llm_calls} AI calls, {result.seconds:.1f}s{tokens} - {result.reason}")

    if result.success:
        e = result.entry

        out("\nPUZZLE")
        out(f"  word:        {e['word']}")
        out(f"  hint:        {e['hint']}")
        out(f"  sentence:    {e['sentence']}")
        out(f"  options:     {', '.join(e['options'])}")
        out(f"  explanation: {e['explanation']}")

        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
