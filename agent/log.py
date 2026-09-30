"""Records every agent run (JSON lines) - the data behind the quality panel."""

import json
import os
from collections import Counter
from datetime import datetime, timezone

import config

_last = None


def clear_last():
    global _last
    _last = None


def last_run():
    return _last


def _summary(observation, limit=220):
    text = json.dumps(observation, ensure_ascii=False)

    return text if len(text) <= limit else text[: limit - 3] + "..."


def run_to_dict(result):
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "theme": result.theme,
        "difficulty": result.difficulty,
        "success": result.success,
        "word": result.entry["word"] if result.entry else None,
        "reason": result.reason,
        "llm_calls": result.llm_calls,
        "tokens": getattr(result, "tokens", 0),
        "seconds": round(result.seconds, 2),
        "steps": [
            {
                "i": s.index,
                "action": s.action,
                "thought": s.thought,
                "observation": _summary(s.observation),
                "ok": s.ok,
                "problems": s.problems,
            }
            for s in result.steps
        ],
    }


def record_run(result, path=None):
    """Remember the run in memory and append it to the log file."""
    global _last

    run = run_to_dict(result)
    _last = run

    if config.AGENT_LOG_ENABLED:
        path = path or config.AGENT_LOG_PATH

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[agent.log] could not write run log: {exc}")

    return run


def load_runs(path=None, limit=500):
    path = path or config.AGENT_LOG_PATH

    runs = []

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    return runs[-limit:]


def summarize_runs(runs):
    if not runs:
        return {"runs": 0}

    ok = [r for r in runs if r.get("success")]

    problems = Counter(
        problem
        for r in runs
        for s in r.get("steps", [])
        for problem in s.get("problems", [])
    )

    failures = Counter(r.get("reason", "unknown") for r in runs if not r.get("success"))

    by_theme = {}

    for r in runs:
        t = by_theme.setdefault(r.get("theme", "?"), {"runs": 0, "success": 0})
        t["runs"] += 1
        t["success"] += int(bool(r.get("success")))

    n = len(runs)

    counted = [r["tokens"] for r in runs if r.get("tokens")]

    return {
        "avg_tokens": round(sum(counted) / len(counted)) if counted else 0,
        "runs": n,
        "success_rate": round(100 * len(ok) / n),
        "avg_steps": round(sum(len(r.get("steps", [])) for r in runs) / n, 1),
        "avg_llm_calls": round(sum(r.get("llm_calls", 0) for r in runs) / n, 1),
        "avg_seconds": round(sum(r.get("seconds", 0) for r in runs) / n, 1),
        "top_rejections": problems.most_common(5),
        "failure_reasons": failures.most_common(3),
        "by_theme": by_theme,
    }
