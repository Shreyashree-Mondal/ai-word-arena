"""
Forgotten PIN? The person who runs the game can clear it.

    python scripts/reset_pin.py Sam

The profile keeps all its progress. The next PIN typed on the Log in tab for that
name becomes its new PIN, so ask the owner to log in straight away.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import store  # noqa: E402


def main(argv=None, out=print):
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="the player whose PIN should be cleared")
    args = parser.parse_args(argv)

    if store.reset_pin(args.name):
        out(f"PIN cleared for {args.name}. Progress is untouched.")
        out("The next PIN typed on the Log in tab for this name becomes the new PIN -")
        out("ask the owner to log in right away.")
        return 0

    out(f"No player called {args.name} was found.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
