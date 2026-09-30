import pytest

import config
from agent import critic as agent_critic
from ai import llm_client
from agent import log as agent_log


@pytest.fixture(autouse=True)
def isolate_runtime_files(monkeypatch, tmp_path):
    """No test may write to the real agent log, puzzle bank or player database."""
    monkeypatch.setattr(config, "AGENT_LOG_PATH", str(tmp_path / "agent_runs.jsonl"))
    monkeypatch.setattr(config, "AUTO_SAVE_AI_PUZZLES", False)
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "aiword.db"))
    monkeypatch.setattr(config, "PIN_HASH_ITERATIONS", 1000)   # fast tests; real value is 600,000

    # Scripts raise these limits; restore them afterwards. Pacing is off unless a test turns it on.
    monkeypatch.setattr(config, "API_TPM_LIMIT", 0)
    monkeypatch.setattr(config, "API_PACING_MAX_WAIT", config.API_PACING_MAX_WAIT)
    monkeypatch.setattr(config, "API_RETRY_MAX_WAIT", config.API_RETRY_MAX_WAIT)

    agent_log.clear_last()
    agent_critic.reset()
    llm_client.reset_pacing()

    yield

    agent_log.clear_last()
    agent_critic.reset()
