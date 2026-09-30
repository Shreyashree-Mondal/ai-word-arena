"""
Extend data/puzzles.json with AI-generated puzzles.

Examples:
    python scripts/build_puzzle_bank.py --target 10
    python scripts/build_puzzle_bank.py --themes "Space,History" --target 15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from ai import llm_client  # noqa: E402
from ai.bank_builder import build_bank  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=10,
                        help="minimum puzzles per theme+difficulty (default 10)")
    parser.add_argument("--themes", default=",".join(config.SECTORS),
                        help="comma-separated themes (default: all)")
    parser.add_argument("--difficulties", default=",".join(config.DIFFICULTY_LEVELS))
    parser.add_argument("--agent", action="store_true",
                        help="use the tool-using agent (slower, self-checking) instead of batch generation")
    args = parser.parse_args()

    if not llm_client.is_available():
        print("AI is not configured. Set your key first, e.g.:")
        print('  export GROQ_API_KEY="your-key"')
        sys.exit(1)

    # Building the bank can take its time: wait out rate limits instead of failing
    config.API_PACING_MAX_WAIT = max(config.API_PACING_MAX_WAIT, 90)
    config.API_RETRY_MAX_WAIT = max(config.API_RETRY_MAX_WAIT, 60)

    themes = [t.strip() for t in args.themes.split(",") if t.strip()]
    difficulties = [d.strip() for d in args.difficulties.split(",") if d.strip()]

    if args.agent:
        from agent.service import build_bank_with_agent

        stats = build_bank_with_agent(themes, difficulties, args.target)

        print()
        print(f"Added {stats['added']} puzzles in {stats['runs']} agent runs "
              f"({stats['failed_runs']} runs did not finish).")
        return

    stats = build_bank(themes, difficulties, args.target)

    print()
    print(f"Added {stats['added']} puzzles in {stats['calls']} AI calls.")

    if stats["rejected"]:
        print("Rejected by validation:", dict(stats["rejected"]))


if __name__ == "__main__":
    main()
