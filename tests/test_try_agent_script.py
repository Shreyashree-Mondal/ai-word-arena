import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from agent_fakes import GOOD, ScriptedLLM, act, submit  # noqa: E402
from try_agent import main  # noqa: E402


def test_script_prints_every_step_and_the_final_puzzle():
    lines = []

    code = main(["Formula 1", "Easy", "--no-critic"],
                llm=ScriptedLLM(act("check_word", "look it up", word="DRIVER"), submit()),
                out=lines.append)

    text = "\n".join(lines)

    assert code == 0
    assert "[1] OK   check_word" in text and "[2] OK   submit_puzzle" in text
    assert "thought: look it up" in text and "PUZZLE" in text and "DRIVER" in text


def test_script_reports_failure_with_a_nonzero_exit_code():
    lines = []

    assert main(["Space", "Hard", "--no-critic"], llm=ScriptedLLM(), out=lines.append) == 2
    assert "LLM unavailable" in "\n".join(lines)


def test_script_explains_how_to_configure_when_ai_is_off():
    lines = []

    assert main(["Space"], out=lines.append) == 1
    assert "GROQ_API_KEY" in "\n".join(lines)


def test_the_script_waits_out_rate_limits_instead_of_failing():
    import config

    config.API_PACING_MAX_WAIT = 12
    config.API_RETRY_MAX_WAIT = 15

    main(["Space", "Easy", "--no-critic"], llm=ScriptedLLM(), out=lambda *_: None)

    assert config.API_PACING_MAX_WAIT >= 90 and config.API_RETRY_MAX_WAIT >= 60


def test_the_script_reports_how_many_tokens_the_run_used():
    from ai import llm_client

    replies = [json.dumps(act("check_word", word="DRIVER")), json.dumps(submit())]

    def metered(prompt, **kwargs):
        llm_client._record_usage({"prompt_tokens": 200, "completion_tokens": 40})
        return replies.pop(0)

    lines = []

    assert main(["Formula 1", "Easy", "--no-critic"], llm=metered, out=lines.append) == 0
    assert "480 tokens" in "\n".join(lines)
