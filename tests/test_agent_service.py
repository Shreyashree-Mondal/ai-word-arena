import json

import pytest

import config
from agent import log as agent_log
from agent import service
from agent.puzzle_agent import AgentResult, Step
from agent_fakes import GARAGE, GOOD, ScriptedLLM, act, submit
from ai.puzzle_bank import load_entries


@pytest.fixture(autouse=True)
def no_critic(monkeypatch):
    monkeypatch.setattr(config, "AGENT_USE_CRITIC", False)


def log_lines():
    with open(config.AGENT_LOG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------- generate_with_agent ----------------

def test_success_returns_an_agent_puzzle_logs_the_run_and_saves_to_the_bank(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AUTO_SAVE_AI_PUZZLES", True)
    bank = tmp_path / "bank.json"

    llm = ScriptedLLM(act("check_word", word="DRIVER"), submit())

    p = service.generate_with_agent("Formula 1", "Easy", llm=llm, bank_path=bank)

    assert p.word == "DRIVER" and p.source == "agent" and p.theme == "Formula 1"
    assert p.options[0] == "DRIVER" and len(p.options) == 4

    assert [e["word"] for e in load_entries(bank)] == ["DRIVER"]

    run = log_lines()[0]
    assert run["success"] is True and run["word"] == "DRIVER" and run["llm_calls"] == 2
    assert [s["action"] for s in run["steps"]] == ["check_word", "submit_puzzle"]
    assert agent_log.last_run()["word"] == "DRIVER"


def test_bank_is_left_alone_when_autosave_is_off(tmp_path):
    bank = tmp_path / "bank.json"

    assert service.generate_with_agent("Formula 1", "Easy", llm=ScriptedLLM(submit()), bank_path=bank)
    assert not bank.exists()


def test_failure_returns_none_and_is_logged():
    p = service.generate_with_agent("Formula 1", "Easy", llm=ScriptedLLM())

    assert p is None
    assert log_lines()[0]["success"] is False and log_lines()[0]["reason"] == "LLM unavailable"


def test_agent_avoids_words_already_in_the_bank(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AUTO_SAVE_AI_PUZZLES", True)
    bank = tmp_path / "bank.json"

    service.generate_with_agent("Formula 1", "Easy", llm=ScriptedLLM(submit()), bank_path=bank)

    second = service.generate_with_agent(
        "Formula 1", "Easy", llm=ScriptedLLM(submit(), submit(GARAGE)), bank_path=bank
    )

    assert second.word == "GARAGE"
    assert sorted(e["word"] for e in load_entries(bank)) == ["DRIVER", "GARAGE"]
    assert log_lines()[1]["steps"][0]["problems"] == ["duplicate word"]


# ---------------- build_bank_with_agent ----------------

def test_bank_builder_tops_up_with_the_agent(tmp_path):
    bank = tmp_path / "bank.json"

    stats = service.build_bank_with_agent(
        ["Formula 1"], ["Easy"], target=2,
        llm=ScriptedLLM(submit(), submit(GARAGE)), bank_path=bank, log_fn=lambda *_: None,
    )

    assert stats["added"] == 2 and stats["runs"] == 2 and stats["failed_runs"] == 0
    assert {e["word"] for e in load_entries(bank)} == {"DRIVER", "GARAGE"}


def test_bank_builder_stops_cleanly_when_the_ai_is_down(tmp_path):
    stats = service.build_bank_with_agent(
        ["Formula 1"], ["Easy"], target=3, llm=ScriptedLLM(),
        bank_path=tmp_path / "b.json", log_fn=lambda *_: None,
    )

    assert stats["aborted"] is True and stats["added"] == 0 and stats["runs"] == 1


def test_bank_builder_keeps_trying_after_a_failed_run(tmp_path):
    bad = act("check_word", word="QQQQQQ")

    # run 1 burns all 8 steps on a bad word; run 2 succeeds straight away
    stats = service.build_bank_with_agent(
        ["Formula 1"], ["Easy"], target=1, max_runs_per_combo=3,
        llm=ScriptedLLM(*([bad] * config.AGENT_MAX_STEPS + [submit()])),
        bank_path=tmp_path / "b.json", log_fn=lambda *_: None,
    )

    assert stats["runs"] == 2 and stats["failed_runs"] == 1 and stats["added"] == 1


def test_bank_builder_gives_up_eventually(tmp_path):
    bad = act("check_word", word="QQQQQQ")

    stats = service.build_bank_with_agent(
        ["Formula 1"], ["Easy"], target=1, max_runs_per_combo=2,
        llm=ScriptedLLM(*([bad] * config.AGENT_MAX_STEPS * 2)),
        bank_path=tmp_path / "b.json", log_fn=lambda *_: None,
    )

    assert stats["runs"] == 2 and stats["failed_runs"] == 2 and stats["added"] == 0


# ---------------- the run log ----------------

def make_result(success, reason="puzzle accepted", theme="Cricket", problems=()):
    steps = [Step(1, "t", "submit_puzzle", {}, {"accepted": success}, success, list(problems))]

    return AgentResult(theme, "Easy", {"word": "WICKET"} if success else None, steps, 3, 2.5, reason)


def test_summary_of_runs():
    agent_log.record_run(make_result(True))
    agent_log.record_run(make_result(True, theme="Music"))
    agent_log.record_run(make_result(False, "step budget exhausted", problems=["hint contains the answer"]))
    agent_log.record_run(make_result(False, "LLM unavailable", problems=["hint contains the answer"]))

    summary = agent_log.summarize_runs(agent_log.load_runs())

    assert summary["runs"] == 4 and summary["success_rate"] == 50
    assert summary["avg_steps"] == 1.0 and summary["avg_llm_calls"] == 3.0 and summary["avg_seconds"] == 2.5
    assert summary["top_rejections"][0] == ("hint contains the answer", 2)
    assert dict(summary["failure_reasons"]) == {"step budget exhausted": 1, "LLM unavailable": 1}
    assert summary["by_theme"]["Cricket"] == {"runs": 3, "success": 1}


def test_log_is_robust_to_missing_files_and_corrupt_lines():
    assert agent_log.load_runs() == [] and agent_log.summarize_runs([]) == {"runs": 0}

    agent_log.record_run(make_result(True))

    with open(config.AGENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("{corrupt line\n")

    assert len(agent_log.load_runs()) == 1


def test_log_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "AGENT_LOG_ENABLED", False)

    agent_log.record_run(make_result(True))

    assert agent_log.load_runs() == [] and agent_log.last_run() is not None


# ---------------- theme-specific words and token reporting ----------------

def test_the_agent_is_shown_only_this_themes_words(tmp_path):
    from agent_fakes import ScriptedLLM as Script
    from ai.puzzle_bank import save_entries

    def entry(word, theme):
        return {"theme": theme, "difficulty": "Easy", "word": word, "hint": "h", "sentence": "_____",
                "options": [word, "A", "B", "C"], "explanation": "e"}

    bank = tmp_path / "bank.json"
    save_entries([entry("SOCCER", "Sports"), entry("ROCKET", "Space")], bank)

    llm = Script(submit(), APPROVE_ALL)
    service.generate_with_agent("Sports", "Easy", llm=llm, bank_path=bank)

    line = next(l for l in llm.prompts[0].splitlines() if l.startswith("Already used"))

    assert "SOCCER" in line and "ROCKET" not in line


APPROVE_ALL = {"options": {}, "fits_theme": True}


def test_theme_matching_ignores_letter_case_and_junk_rows():
    rows = [{"theme": "sports", "word": "SOCCER"}, {"theme": " SPORTS ", "word": "TENNIS"},
            {"theme": "Space", "word": "ROCKET"}, {"theme": "Sports"}, {}]

    assert service._theme_words(rows, "Sports") == ["SOCCER", "TENNIS"]


def test_tokens_are_logged_and_averaged():
    ok = make_result(True)
    ok.tokens = 3000
    other = make_result(True)
    other.tokens = 1000
    unmetered = make_result(True)                       # e.g. a run that used a fake LLM: not counted

    for r in (ok, other, unmetered):
        agent_log.record_run(r)

    runs = agent_log.load_runs()

    assert [r["tokens"] for r in runs] == [3000, 1000, 0]
    assert agent_log.summarize_runs(runs)["avg_tokens"] == 2000


def test_older_log_lines_without_tokens_are_still_summarised():
    old = {"theme": "Space", "difficulty": "Easy", "success": True, "steps": [], "llm_calls": 2, "seconds": 1.0}

    assert agent_log.summarize_runs([old])["avg_tokens"] == 0
