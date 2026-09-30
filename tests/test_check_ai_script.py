import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_ai  # noqa: E402
import config  # noqa: E402
from ai import llm_client  # noqa: E402


class Resp:
    def __init__(self, status=200, data=None):
        self.status_code, self.ok, self._data = status, status < 400, data or {}

    def json(self):
        return self._data


MODELS = {"data": [{"id": "openai/gpt-oss-20b"}, {"id": "whisper-large-v3"}, {"id": "qwen/qwen3.8-27b"},
                   {"id": "meta-llama/llama-prompt-guard-2-22m"}]}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(config, "AI_PROVIDER", "api")
    monkeypatch.setattr(config, "API_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_API_KEY", "secret-key-123")


def run():
    lines = []
    code = check_ai.main(out=lines.append)
    return code, "\n".join(lines)


def test_reports_success_and_lists_only_chat_models(api, monkeypatch):
    monkeypatch.setattr(check_ai.requests, "get", lambda *a, **k: Resp(200, MODELS))
    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: "ready")

    code, text = run()

    assert code == 0 and "All good" in text
    assert "openai/gpt-oss-20b" in text and "qwen/qwen3.8-27b" in text
    assert "whisper" not in text and "prompt-guard" not in text        # not chat models
    assert "secret-key-123" not in text                                # the key is never printed


def test_tells_the_user_when_the_configured_model_is_retired(api, monkeypatch):
    monkeypatch.setattr(config, "API_MODEL", "llama-3.1-8b-instant")
    monkeypatch.setattr(check_ai.requests, "get", lambda *a, **k: Resp(200, MODELS))

    code, text = run()

    assert code == 2 and "NOT in that list" in text and "AIWORD_API_MODEL" in text


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_key_is_reported_clearly(api, monkeypatch, status):
    monkeypatch.setattr(check_ai.requests, "get", lambda *a, **k: Resp(status))

    code, text = run()

    assert code == 2 and "key was rejected" in text and "secret-key-123" not in text


def test_a_wrong_address_is_reported(api, monkeypatch):
    monkeypatch.setattr(check_ai.requests, "get", lambda *a, **k: Resp(404))

    code, text = run()

    assert code == 2 and "404" in text and "address" in text


def test_no_internet_is_reported_without_a_traceback(api, monkeypatch):
    def offline(*a, **k):
        raise check_ai.requests.ConnectionError("no route")

    monkeypatch.setattr(check_ai.requests, "get", offline)

    code, text = run()

    assert code == 2 and "Could not reach the service" in text


def test_a_failing_test_request_is_reported(api, monkeypatch):
    monkeypatch.setattr(check_ai.requests, "get", lambda *a, **k: Resp(200, MODELS))
    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: None)

    code, text = run()

    assert code == 3 and "test request failed" in text


def test_explains_what_to_do_when_no_key_is_set(monkeypatch):
    monkeypatch.setattr(config, "AI_PROVIDER", "api")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("AIWORD_API_KEY", raising=False)

    code, text = run()

    assert code == 1 and "No API key" in text and "set GROQ_API_KEY" in text


def test_explains_what_to_do_when_the_api_is_not_selected(monkeypatch):
    monkeypatch.setattr(config, "AI_PROVIDER", "none")

    code, text = run()

    assert code == 1 and "not selected" in text


def test_warns_when_the_reviewer_model_is_not_available_but_still_succeeds(api, monkeypatch):
    monkeypatch.setattr(config, "CRITIC_MODEL", "some-reviewer-you-do-not-have")
    monkeypatch.setattr(check_ai.requests, "get", lambda *a, **k: Resp(200, MODELS))
    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: "ready")

    code, text = run()

    assert code == 0 and "reviewer model 'some-reviewer-you-do-not-have' is not available" in text
    assert "AIWORD_CRITIC_MODEL" in text


def test_no_warning_when_the_reviewer_model_is_available(api, monkeypatch):
    monkeypatch.setattr(config, "CRITIC_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(check_ai.requests, "get", lambda *a, **k: Resp(200, MODELS))
    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: "ready")

    assert "reviewer model" not in run()[1]


# ---------------- --limits: measure every model's rate limits ----------------

class Measured(Resp):
    def __init__(self, status=200, headers=None, data=None):
        super().__init__(status, data)
        self.headers = headers or {}
        self.text = ""


def limit_headers(per_minute, per_day, left):
    return {"x-ratelimit-limit-tokens": str(per_minute), "x-ratelimit-limit-requests": str(per_day),
            "x-ratelimit-remaining-tokens": str(left)}


CHAT = {"data": [{"id": "openai/gpt-oss-20b"}, {"id": "openai/gpt-oss-120b"}, {"id": "whisper-large-v3"}]}


def run_with(argv, monkeypatch, post):
    monkeypatch.setattr(check_ai.requests, "get", lambda *a, **k: Resp(200, CHAT))
    monkeypatch.setattr(check_ai.requests, "post", post)
    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: "ready")

    lines = []

    return check_ai.main(argv, out=lines.append), "\n".join(lines)


def test_limits_are_not_measured_unless_asked(api, monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("no measuring requests without --limits")

    code, text = run_with([], monkeypatch, must_not_run)

    assert code == 0 and "Rate limits" not in text


def test_limits_lists_every_chat_model_and_names_the_best_one(api, monkeypatch):
    answers = {"openai/gpt-oss-20b": limit_headers(8000, 1000, 7990), "openai/gpt-oss-120b": limit_headers(20000, 500, 19990)}

    code, text = run_with(["--limits"], monkeypatch, lambda url, headers, json, timeout: Measured(200, answers[json["model"]]))

    assert code == 0
    assert "openai/gpt-oss-20b" in text and "8000" in text and "1000 requests/day" in text
    assert "openai/gpt-oss-120b" in text and "20000" in text
    assert "whisper" not in text                                                    # not a chat model
    assert "Highest tokens-per-minute limit: openai/gpt-oss-120b (20000)" in text
    assert "secret-key-123" not in text


def test_a_model_that_does_not_report_limits_is_shown_gracefully(api, monkeypatch):
    code, text = run_with(["--limits"], monkeypatch, lambda *a, **k: Measured(200, {}))

    assert code == 0 and "not reported" in text and "Highest tokens-per-minute" not in text


def test_a_model_that_cannot_be_measured_does_not_stop_the_check(api, monkeypatch):
    def flaky(url, headers, json, timeout):
        if json["model"].endswith("120b"):
            return Measured(429, {}, {"error": {"message": "Rate limit reached"}})

        return Measured(200, limit_headers(8000, 1000, 7000))

    code, text = run_with(["--limits"], monkeypatch, flaky)

    assert code == 0 and "could not be measured" in text and "openai/gpt-oss-20b" in text


def test_a_network_error_while_measuring_is_reported_not_raised(api, monkeypatch):
    def offline(*a, **k):
        raise check_ai.requests.ConnectionError("no route")

    code, text = run_with(["--limits"], monkeypatch, offline)

    assert code == 0 and "could not be measured: ConnectionError" in text


def test_models_tied_for_the_highest_limit_are_reported_as_tied_not_as_one_winner(api, monkeypatch):
    same = limit_headers(8000, 1000, 7990)

    code, text = run_with(["--limits"], monkeypatch, lambda url, headers, json, timeout: Measured(200, same))

    assert code == 0
    assert "shared equally by openai/gpt-oss-120b, openai/gpt-oss-20b" in text and "8000" in text
    assert "Highest tokens-per-minute limit: openai" not in text                 # no false single winner
