"""
The Puzzle Master agent.

An LLM runs in a loop: each turn it replies with ONE JSON action, code
executes the chosen tool, and the result is fed back to the model. The loop
ends when a submitted puzzle passes the hard validation gate, or when the
step / time budget runs out.

Safety properties (all covered by tests):
  * the model never validates its own work - code does
  * a puzzle is accepted only through ToolBox.validate_submission
  * the optional critic can only reject; it may send the agent back a limited
    number of times, and if it still rejects the run FAILS (the game then uses
    another source) - a puzzle the critic rejected is never accepted
  * malformed replies, unknown tools and bad arguments cost a step, nothing more
  * if the LLM is unavailable the run stops immediately
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import config
from agent import critic as critic_module
from agent.tools import TOOL_DOCS, ToolBox
from ai import llm_client


@dataclass
class Step:
    index: int
    thought: str
    action: str
    args: dict
    observation: dict
    ok: bool
    problems: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    theme: str
    difficulty: str
    entry: Optional[dict]
    steps: List[Step]
    llm_calls: int
    seconds: float
    reason: str
    tokens: int = 0          # tokens the provider reported for this run (approximate under load)

    @property
    def success(self):
        return self.entry is not None


def normalise_tool_name(name):
    """
    Models sometimes namespace a tool: "tool.check_word", "functions.check_word".
    Strip the prefix when what remains is one of OUR tools; leave anything else
    untouched so genuinely unknown tools are still reported as unknown.
    """
    plain = name.strip()

    for separator in (".", "/", ":"):
        if separator in plain:
            tail = plain.rsplit(separator, 1)[1].strip().lower()

            if tail in TOOL_DOCS:
                return tail

    return plain.lower() if plain.lower() in TOOL_DOCS else plain


_CALL_MARKER = re.compile(r"to=\s*([A-Za-z_][\w.:/-]*)")


def _tool_named_before(prefix):
    """The tool named in `to=functions.check_word`, if it is one of ours."""
    found = _CALL_MARKER.findall(prefix)

    if not found:
        return None

    name = normalise_tool_name(found[-1])

    return name if name in TOOL_DOCS else None


def parse_action(text):
    """
    Returns (thought, action, args) or None.
    Tolerates code fences, chatter, several JSON objects in one reply, and a
    submit_puzzle whose fields were written at the top level.
    """
    if not text:
        return None

    text = text.replace("```json", "").replace("```", "")

    start = text.find("{")

    if start == -1:
        return None

    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    # A model's built-in call syntax: `... to=functions.check_word <|message|>{"word": "x"}`.
    # The tool name sits BEFORE the JSON, which then holds only the arguments.
    if "action" not in data and "name" not in data:
        named = _tool_named_before(text[:start])

        if named:
            return "", named, data

    # Our format is {"action", "args"}; many models use their built-in
    # tool-call format {"name", "arguments"} - accept both.
    action = data.get("action")

    if not isinstance(action, str):
        action = data.get("name")

    if not isinstance(action, str) or not action.strip():
        return None

    action = normalise_tool_name(action)

    args = data.get("args")

    if not isinstance(args, dict):
        args = data.get("arguments")

    if isinstance(args, str):  # some models send the arguments as JSON text
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = None

    if not isinstance(args, dict):
        args = (
            {
                k: v
                for k, v in data.items()
                if k not in ("action", "name", "thought", "args", "arguments")
            }
            if action == "submit_puzzle"
            else {}
        )

    return str(data.get("thought", ""))[:300], action, args


# ---------------------------------------
# Keeping the prompt small (it is re-sent on every step and counts against the
# provider's tokens-per-minute limit)
# ---------------------------------------

RECENT_STEPS_IN_FULL = 3
MAX_OLDER_LINES = 6            # older steps beyond these are only counted, not listed
_NOISY_KEYS = {"zipf", "part_of_speech", "target", "reason_code"}


def _lean(observation):
    if not isinstance(observation, dict):
        return observation

    lean = {k: v for k, v in observation.items() if k not in _NOISY_KEYS}

    if isinstance(lean.get("definition"), str):
        lean["definition"] = lean["definition"][:60]

    return lean


def _one_line(observation):
    """A very short summary of an older step's result."""
    if not isinstance(observation, dict):
        return str(observation)[:80]

    if "error" in observation:
        return f"error: {str(observation['error'])[:80]}"

    if "accepted" in observation:
        problems = "; ".join(observation.get("problems", []))[:100]

        return "accepted" if observation["accepted"] else f"rejected: {problems}"

    if isinstance(observation.get("usable"), list):
        good = ", ".join(w["word"] for w in observation["usable"]) or "none"
        bad = ", ".join(f"{w['word']} ({w['why']})" for w in observation.get("rejected", []))

        return f"usable: {good}; rejected: {bad}"[:160]

    if "usable" in observation:
        note = observation.get("advice") or observation.get("reason") or ""

        return "usable" if observation["usable"] else f"not usable: {note}"[:100]

    return json.dumps(_lean(observation), ensure_ascii=False)[:120]


def _history_text(steps):
    """Older steps as one line each; only the most recent steps in full."""
    if not steps:
        return ""

    older, recent = steps[:-RECENT_STEPS_IN_FULL], steps[-RECENT_STEPS_IN_FULL:]

    text = "\nEarlier steps:\n" if older else ""

    if len(older) > MAX_OLDER_LINES:
        text += f"  ({len(older) - MAX_OLDER_LINES} earlier steps not shown)\n"
        older = older[-MAX_OLDER_LINES:]

    for s in older:
        text += f"  {s.index}. {s.action} -> {_one_line(s.observation)}\n"

    for s in recent:
        call = json.dumps(
            {"thought": s.thought[:150], "action": s.action, "args": s.args},
            ensure_ascii=False,
        )[:350]

        obs = json.dumps(_lean(s.observation), ensure_ascii=False)[:350]

        text += f"\nStep {s.index}: {call}\nObservation {s.index}: {obs}\n"

    return text


class PuzzleAgent:
    def __init__(
        self,
        llm=None,
        max_steps=None,
        time_budget=None,
        use_critic=None,
        clock=time.monotonic,
    ):
        self.llm = llm or llm_client.generate_text
        self.max_steps = max_steps or config.AGENT_MAX_STEPS
        self.time_budget = time_budget or config.AGENT_TIME_BUDGET
        self.use_critic = config.AGENT_USE_CRITIC if use_critic is None else use_critic
        self.clock = clock

    # ---------------- prompt ----------------

    def _prompt(self, theme, difficulty, toolbox, steps):
        tool_lines = "\n".join(f"- {name} {doc}" for name, doc in TOOL_DOCS.items())

        # Words used in THIS theme first: they are the ones the model is likely to reach for.
        pool = toolbox.avoid if toolbox.avoid is not None else sorted(set(toolbox.taken))

        used = ", ".join(pool[:40]) or "none"

        history = _history_text(steps)

        return f"""You are the Puzzle Master agent for a vocabulary game.
Goal: create ONE puzzle about the theme "{theme}" at {difficulty} difficulty.

A puzzle has:
- word: the answer. EXACTLY 6 letters, a real English base-form word (no plurals, no verb forms), connected to the theme.
- hint: ONE sentence clue that does not contain the word or any form of it.
- sentence: one sentence with the word replaced by _____ (five underscores).
- distractors: EXACTLY 3 real English words that do NOT correctly complete the sentence.
- explanation: one sentence saying why the answer is right.

Tools (results come from a real dictionary - trust them over your own memory):
{tool_lines}

A GOOD puzzle has exactly ONE fair answer:
- The sentence needs a specific clue that ONLY the answer satisfies (what it does, looks like or is used for). "The ___ was called off due to rain" is BAD: soccer, tennis and hockey all fit. GOOD: "Players hit a shuttlecock over a high net in a game of ___".
- Put each wrong option into the blank; if it could ALSO be right, fix the sentence or the option.
- Wrong options: plausible words that clearly do NOT fit the clue.

Already used (never reuse, plurals included): {used}

Reply with EXACTLY ONE JSON object and nothing else:
{{"thought": "<short>", "action": "<tool name>", "args": {{...}}}}

Suggested approach: brainstorm 6-8 candidate words and check_words them ALL in one call; write the hint and sentence; then submit_puzzle.
{history}
Your next JSON action:"""

    # ---------------- run ----------------

    def run(self, theme, difficulty, used_words=(), bank_words=(), theme_words=None):
        toolbox = ToolBox(theme, difficulty, used_words, bank_words)

        if theme_words is not None:
            toolbox.avoid = list(
                dict.fromkeys([w.upper() for w in theme_words] + [w.upper() for w in used_words])
            )

        usage_before = llm_client.usage_snapshot()

        steps, llm_calls, critic_rejections = [], 0, 0

        start = self.clock()

        def finish(entry, reason):
            return AgentResult(
                theme=theme,
                difficulty=difficulty,
                entry=entry,
                steps=steps,
                llm_calls=llm_calls,
                seconds=self.clock() - start,
                reason=reason,
                tokens=llm_client.usage_since(usage_before)["total_tokens"],
            )

        for index in range(1, self.max_steps + 1):
            if self.clock() - start > self.time_budget:
                return finish(None, "time budget exhausted")

            text = self.llm(
                self._prompt(theme, difficulty, toolbox, steps),
                max_tokens=config.AGENT_MAX_TOKENS,
                temperature=0.4,
                timeout=config.LIVE_AI_TIMEOUT,
            )

            llm_calls += 1

            if text is None:
                return finish(None, "LLM unavailable")

            parsed = parse_action(text)

            if parsed is None:
                steps.append(
                    Step(
                        index, "", "invalid_reply", {},
                        {"error": "Reply with exactly one JSON object with keys "
                                  "thought, action and args."},
                        ok=False,
                        problems=["invalid reply format"],
                    )
                )
                continue

            thought, action, args = parsed

            # ---- final submission: hard gate, then optional critic ----
            if action == "submit_puzzle":
                verdict, entry = toolbox.validate_submission(args)

                give_up = False

                if entry is not None and self.use_critic:
                    review = critic_module.review(entry, theme, self.llm)

                    if review["called"]:
                        llm_calls += 1

                    if not review["reject"]:
                        verdict = {
                            **verdict,
                            "review": review["reason"],
                            "reviewer": review.get("model"),
                        }

                    if review["reject"]:
                        critic_rejections += 1
                        give_up = critic_rejections > config.AGENT_MAX_CRITIC_REJECTIONS

                        verdict = {
                            "accepted": False,
                            "problems": [review["reason"]],
                            "reviewer": review.get("model"),
                        }
                        entry = None

                steps.append(
                    Step(
                        index, thought, action, args, verdict,
                        ok=verdict["accepted"],
                        problems=list(verdict["problems"]),
                    )
                )

                if entry is not None:
                    return finish(entry, "puzzle accepted")

                if give_up:
                    return finish(None, "critic kept rejecting the puzzle")

                continue

            # ---- ordinary tool call ----
            observation = toolbox.call(action, args)

            failed = "error" in observation

            steps.append(
                Step(
                    index, thought, action, args, observation,
                    ok=not failed,
                    problems=[observation["error"]] if failed else [],
                )
            )

        return finish(None, "step budget exhausted")
