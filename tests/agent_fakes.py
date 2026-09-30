"""Shared fakes for the agent tests: a scripted LLM and ready-made puzzles."""

import json

GOOD = dict(
    word="DRIVER",
    hint="The person who steers a racing car around the track.",
    sentence="The _____ took the final corner at full speed.",
    distractors=["MECHANIC", "COMMENTATOR", "SPECTATOR"],
    explanation="A driver controls the car.",
)

GARAGE = dict(
    word="GARAGE",
    hint="The area where teams repair and prepare the cars.",
    sentence="The mechanics pushed the car back into the _____ after the race.",
    distractors=["TROPHY", "GRANDSTAND", "SPONSOR"],
    explanation="A garage is where the cars are serviced.",
)


def act(action, thought="", **args):
    return {"thought": thought, "action": action, "args": args}


def submit(puzzle=GOOD, **override):
    return act("submit_puzzle", "submitting", **{**puzzle, **override})


class ScriptedLLM:
    """Plays a fixed transcript. When the script runs out it behaves like a dead provider."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.kwargs = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)

        if not self.replies:
            return None

        reply = self.replies.pop(0)

        return reply if isinstance(reply, str) else json.dumps(reply)


