import json

import pytest

import config
from ai import llm_client


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status
        self.ok = status < 400
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def chat_reply(content="Hello!", **message):
    return {"choices": [{"message": {"role": "assistant", "content": content, **message}}]}


def groq_error(message, code=None, failed_generation=None):
    error = {"message": message, "type": "invalid_request_error"}

    if code:
        error["code"] = code

    if failed_generation is not None:
        error["failed_generation"] = failed_generation

    return {"error": error}


@pytest.fixture
def api_mode(monkeypatch):
    monkeypatch.setattr(config, "AI_PROVIDER", "api")
    monkeypatch.setattr(config, "API_MODEL", "some-plain-model")
    monkeypatch.setattr(config, "API_EXTRA_TOKENS", "auto")
    monkeypatch.setattr(config, "API_REASONING_EFFORT", "auto")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("AIWORD_API_KEY", raising=False)


def capture(monkeypatch, response):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json, timeout=timeout)
        return response

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    return seen


# ---------------- availability ----------------

def test_off_by_default_returns_none(monkeypatch):
    monkeypatch.setattr(config, "AI_PROVIDER", "none")

    assert llm_client.is_available() is False
    assert llm_client.generate_text("hi") is None


def test_missing_key_means_unavailable(monkeypatch):
    monkeypatch.setattr(config, "AI_PROVIDER", "api")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("AIWORD_API_KEY", raising=False)

    assert llm_client.is_available() is False
    assert llm_client.generate_text("hi") is None


# ---------------- a normal call ----------------

def test_api_call_sends_key_and_strips_thinking(api_mode, monkeypatch):
    seen = capture(monkeypatch, FakeResponse(chat_reply("<think>x</think>Hello!")))

    assert llm_client.generate_text("hi") == "Hello!"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["body"]["messages"][0]["content"] == "hi"


def test_plain_models_get_the_requested_token_limit_and_no_reasoning_setting(api_mode, monkeypatch):
    seen = capture(monkeypatch, FakeResponse(chat_reply()))

    llm_client.generate_text("hi", max_tokens=200)

    assert seen["body"]["max_tokens"] == 200 and "reasoning_effort" not in seen["body"]


# ---------------- reasoning models ----------------

def test_reasoning_models_get_extra_room_and_low_effort(api_mode, monkeypatch):
    monkeypatch.setattr(config, "API_MODEL", "openai/gpt-oss-20b")
    seen = capture(monkeypatch, FakeResponse(chat_reply()))

    llm_client.generate_text("hi", max_tokens=200)

    assert seen["body"]["max_tokens"] == 700 and seen["body"]["reasoning_effort"] == "low"


def test_other_reasoning_families_get_room_but_no_effort_setting(api_mode, monkeypatch):
    monkeypatch.setattr(config, "API_MODEL", "qwen/qwen3.8-27b")
    seen = capture(monkeypatch, FakeResponse(chat_reply()))

    llm_client.generate_text("hi", max_tokens=200)

    assert seen["body"]["max_tokens"] == 700 and "reasoning_effort" not in seen["body"]


@pytest.mark.parametrize("extra,effort,expected_tokens,expected_effort", [
    ("0", "off", 200, None),
    ("300", "high", 500, "high"),
    ("nonsense", "none", 200, None),
    ("-5", "", 200, None),
])
def test_token_room_and_effort_can_be_overridden(api_mode, monkeypatch, extra, effort, expected_tokens, expected_effort):
    monkeypatch.setattr(config, "API_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(config, "API_EXTRA_TOKENS", extra)
    monkeypatch.setattr(config, "API_REASONING_EFFORT", effort)
    seen = capture(monkeypatch, FakeResponse(chat_reply()))

    llm_client.generate_text("hi", max_tokens=200)

    assert seen["body"]["max_tokens"] == expected_tokens
    assert seen["body"].get("reasoning_effort") == expected_effort


def test_the_hidden_reasoning_text_is_never_used_as_the_answer(api_mode, monkeypatch):
    capture(monkeypatch, FakeResponse(chat_reply("Hi!", reasoning="The user said hi, so I should...")))

    assert llm_client.generate_text("hi") == "Hi!"


# ---------------- errors ----------------

def test_errors_never_raise_and_show_the_reason_given_by_the_service(api_mode, monkeypatch, capsys):
    capture(monkeypatch, FakeResponse(groq_error("The model `old-one` does not exist"), status=404))

    assert llm_client.generate_text("hi") is None

    printed = capsys.readouterr().out

    assert "404" in printed and "does not exist" in printed and "test-key" not in printed


def test_rate_limits_are_reported_not_raised(api_mode, monkeypatch, capsys):
    capture(monkeypatch, FakeResponse(groq_error("Rate limit reached"), status=429))

    assert llm_client.generate_text("hi") is None
    assert "Rate limit reached" in capsys.readouterr().out


def test_a_non_json_error_body_is_still_handled(api_mode, monkeypatch, capsys):
    class Html:
        status_code, ok, text = 502, False, "<html>Bad gateway</html>"

        def json(self):
            raise ValueError("not json")

    capture(monkeypatch, Html())

    assert llm_client.generate_text("hi") is None
    assert "502" in capsys.readouterr().out


# ---------------- models that answer in their built-in tool-call format ----------------

def test_a_refused_tool_call_is_recovered_from_the_error(api_mode, monkeypatch):
    """The exact response a real user got from Groq with openai/gpt-oss-20b."""
    real = groq_error(
        "Tool choice is none, but model called a tool",
        code="tool_use_failed",
        failed_generation='{"name": "check_word", "arguments": {"word":"helmet"}}',
    )
    capture(monkeypatch, FakeResponse(real, status=400))

    assert llm_client.generate_text("hi") == '{"name": "check_word", "arguments": {"word":"helmet"}}'


def test_other_400_errors_are_not_mistaken_for_tool_calls(api_mode, monkeypatch, capsys):
    capture(monkeypatch, FakeResponse(groq_error("max_tokens is too large"), status=400))

    assert llm_client.generate_text("hi") is None
    assert "max_tokens is too large" in capsys.readouterr().out


def test_unusable_model_output_is_one_bad_reply_not_an_outage(api_mode, monkeypatch, capsys):
    """Groq refused because the MODEL garbled its answer - the service itself is fine."""
    body = groq_error("x", code="tool_use_failed", failed_generation="")
    capture(monkeypatch, FakeResponse(body, status=400))

    assert llm_client.generate_text("hi") == ""                    # empty reply, NOT None (= unavailable)
    assert "unusable output" in capsys.readouterr().out


def test_a_successful_reply_that_only_contains_a_tool_call_is_converted(api_mode, monkeypatch):
    reply = chat_reply("", tool_calls=[{"function": {"name": "check_word", "arguments": '{"word": "helmet"}'}}])
    capture(monkeypatch, FakeResponse(reply))

    assert json.loads(llm_client.generate_text("hi")) == {"name": "check_word", "arguments": '{"word": "helmet"}'}


def test_an_empty_reply_is_an_empty_string_not_a_crash(api_mode, monkeypatch):
    capture(monkeypatch, FakeResponse(chat_reply("")))

    assert llm_client.generate_text("hi") == ""


# ---------------- rate limits ----------------

class Limited(FakeResponse):
    def __init__(self, message="Rate limit reached", retry_after=None):
        super().__init__(groq_error(message), status=429)
        self.headers = {} if retry_after is None else {"retry-after": str(retry_after)}


@pytest.fixture
def sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(llm_client.time, "sleep", slept.append)
    return slept


def sequence(monkeypatch, *responses):
    queue = list(responses)
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return queue.pop(0)

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    return calls


def test_a_short_rate_limit_wait_is_honoured_once_and_then_it_works(api_mode, monkeypatch, sleeps):
    calls = sequence(monkeypatch, Limited(retry_after=2), FakeResponse(chat_reply("Recovered!")))

    assert llm_client.generate_text("hi") == "Recovered!"
    assert len(calls) == 2 and sleeps == [2.25]


@pytest.mark.parametrize("message,expected", [
    ("Rate limit reached. Please try again in 1.5s.", 1.75),
    ("Rate limit reached. Please try again in 250ms.", 0.5),
    ("Rate limit reached for model X. Please try again in 3s. Need more tokens?", 3.25),
])
def test_the_wait_is_read_from_the_error_message_when_there_is_no_header(api_mode, monkeypatch, sleeps, message, expected):
    sequence(monkeypatch, Limited(message), FakeResponse(chat_reply("ok")))

    assert llm_client.generate_text("hi") == "ok"
    assert sleeps == [pytest.approx(expected)]


@pytest.mark.parametrize("message", [
    "Rate limit reached. Please try again in 1h2m3s.",          # a daily limit
    "Rate limit reached. Please try again in 20s.",             # longer than the allowed wait
    "Rate limit reached.",                                      # no idea how long
])
def test_long_or_unknown_waits_are_reported_not_slept_through(api_mode, monkeypatch, sleeps, capsys, message):
    calls = sequence(monkeypatch, Limited(message))

    assert llm_client.generate_text("hi") is None
    assert len(calls) == 1 and sleeps == []
    assert "429" in capsys.readouterr().out


def test_it_retries_only_once_never_in_a_loop(api_mode, monkeypatch, sleeps):
    calls = sequence(monkeypatch, Limited(retry_after=1), Limited(retry_after=1), Limited(retry_after=1))

    assert llm_client.generate_text("hi") is None
    assert len(calls) == 2 and len(sleeps) == 1


def test_other_errors_are_not_retried(api_mode, monkeypatch, sleeps):
    calls = sequence(monkeypatch, FakeResponse(groq_error("server error"), status=500))

    assert llm_client.generate_text("hi") is None
    assert len(calls) == 1 and sleeps == []


def test_the_wait_limit_can_be_changed(api_mode, monkeypatch, sleeps):
    monkeypatch.setattr(config, "API_RETRY_MAX_WAIT", 30)
    sequence(monkeypatch, Limited(retry_after=20), FakeResponse(chat_reply("ok")))

    assert llm_client.generate_text("hi") == "ok" and sleeps == [20.25]


# ---------------- choosing the model per call ----------------

def test_a_call_can_name_its_own_model_and_that_model_decides_the_token_room(api_mode, monkeypatch):
    seen = capture(monkeypatch, FakeResponse(chat_reply()))

    llm_client.generate_text("hi", max_tokens=200, model="openai/gpt-oss-120b")

    assert seen["body"]["model"] == "openai/gpt-oss-120b"
    assert seen["body"]["max_tokens"] == 700 and seen["body"]["reasoning_effort"] == "low"

    llm_client.generate_text("hi", max_tokens=200)                       # the default model is a plain one here

    assert seen["body"]["model"] == "some-plain-model"
    assert seen["body"]["max_tokens"] == 200 and "reasoning_effort" not in seen["body"]


# ---------------- "Parsing failed": the error a real user hit ----------------

PARSING_FAILED = ("Parsing failed. The model generated output that could not be parsed. "
                  "Please adjust your prompt. See 'failed_generation' for more details.")


def test_any_400_that_carries_the_models_attempt_is_recovered_whatever_its_code(api_mode, monkeypatch):
    attempt = '<|channel|>commentary to=functions.check_word<|message|>{"word":"wicket"}'

    capture(monkeypatch, FakeResponse(groq_error(PARSING_FAILED, code="something_else", failed_generation=attempt), status=400))

    assert llm_client.generate_text("hi") == attempt


def test_parsing_failed_without_the_attempt_is_a_bad_reply_not_an_outage(api_mode, monkeypatch, capsys):
    capture(monkeypatch, FakeResponse(groq_error(PARSING_FAILED), status=400))

    assert llm_client.generate_text("hi") == ""
    assert "could not be parsed" in capsys.readouterr().out


@pytest.mark.parametrize("message", ["max_tokens is too large", "model does not exist", "invalid request"])
def test_genuine_request_problems_are_still_reported_as_unavailable(api_mode, monkeypatch, message):
    capture(monkeypatch, FakeResponse(groq_error(message), status=400))

    assert llm_client.generate_text("hi") is None


# ---------------- pacing: stay under tokens-per-minute ----------------

class FakeClock:
    """time.monotonic / time.sleep that never really wait."""

    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 2))
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(llm_client.time, "monotonic", c.monotonic)
    monkeypatch.setattr(llm_client.time, "sleep", c.sleep)
    monkeypatch.setattr(config, "API_TPM_LIMIT", 3000)
    monkeypatch.setattr(config, "API_PACING_MAX_WAIT", 12)
    llm_client.reset_pacing()
    return c


PROMPT = "x" * 400            # 100 estimated tokens


def test_the_estimate_is_prompt_tokens_plus_the_reply_room():
    assert llm_client._estimate_tokens("x" * 4000, 800) == 1800


def test_requests_within_the_budget_are_not_delayed(clock):
    for _ in range(3):
        llm_client._wait_for_budget(llm_client._estimate_tokens(PROMPT, 900))     # 1,000 each

    assert clock.sleeps == []


def test_a_request_that_would_exceed_the_minute_waits_until_old_usage_expires(clock):
    for _ in range(3):
        llm_client._wait_for_budget(1000)                 # fills the 3,000 budget at t=1000

    clock.now += 45                                       # 45 s later the window has 15 s left

    llm_client._wait_for_budget(1000)

    assert len(clock.sleeps) == 1 and 10 < clock.sleeps[0] <= 12       # waits ~15 s but only up to the cap


def test_the_wait_is_capped_and_the_request_is_then_sent_anyway(clock):
    for _ in range(3):
        llm_client._wait_for_budget(1000)

    llm_client._wait_for_budget(1000)                     # a full minute would be needed

    assert sum(clock.sleeps) <= config.API_PACING_MAX_WAIT + 0.01


def test_usage_older_than_a_minute_no_longer_counts(clock):
    for _ in range(3):
        llm_client._wait_for_budget(1000)

    clock.now += 61

    llm_client._wait_for_budget(1000)

    assert clock.sleeps == []


def test_a_single_huge_request_is_never_blocked_forever(clock):
    llm_client._wait_for_budget(50_000)                   # bigger than the whole limit

    assert clock.sleeps == []


def test_pacing_can_be_switched_off(clock, monkeypatch):
    monkeypatch.setattr(config, "API_TPM_LIMIT", 0)

    for _ in range(50):
        llm_client._wait_for_budget(5000)

    assert clock.sleeps == []


def test_real_calls_are_paced_end_to_end(api_mode, monkeypatch, clock):
    capture(monkeypatch, FakeResponse(chat_reply()))

    for _ in range(5):                                    # each reserves ~1,000 tokens against a 3,000 budget
        llm_client.generate_text("x" * 400, max_tokens=900)

    assert clock.sleeps                                                # it did slow down
    assert max(clock.sleeps) <= config.API_PACING_MAX_WAIT + 0.01     # but never more than the cap per request


def test_groq_gets_the_8000_default_and_other_providers_get_none():
    import importlib

    import config as cfg

    try:
        import os

        os.environ.pop("AIWORD_API_TPM", None)
        os.environ.pop("AIWORD_API_URL", None)
        assert importlib.reload(cfg).API_TPM_LIMIT == 8000

        os.environ["AIWORD_API_URL"] = "https://openrouter.ai/api/v1/chat/completions"
        assert importlib.reload(cfg).API_TPM_LIMIT == 0

        os.environ["AIWORD_API_TPM"] = "5000"
        assert importlib.reload(cfg).API_TPM_LIMIT == 5000
    finally:
        os.environ.pop("AIWORD_API_TPM", None)
        os.environ.pop("AIWORD_API_URL", None)
        importlib.reload(cfg)


# ---------------- counting the tokens the provider actually reports ----------------

def usage_reply(prompt=100, completion=20):
    return {**chat_reply("ok"), "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}


def test_reported_usage_is_added_up_between_two_snapshots(api_mode, monkeypatch):
    sequence(monkeypatch, FakeResponse(usage_reply(100, 20)), FakeResponse(usage_reply(300, 50)))

    before = llm_client.usage_snapshot()

    llm_client.generate_text("hi")
    llm_client.generate_text("hi")

    used = llm_client.usage_since(before)

    assert used == {"calls": 2, "prompt_tokens": 400, "completion_tokens": 70, "total_tokens": 470}


@pytest.mark.parametrize("usage", [None, {}, "text", {"prompt_tokens": "many"}, {"prompt_tokens": None, "completion_tokens": None}])
def test_missing_or_malformed_usage_never_breaks_a_call(api_mode, monkeypatch, usage):
    reply = chat_reply("fine")
    reply["usage"] = usage
    capture(monkeypatch, FakeResponse(reply))

    before = llm_client.usage_snapshot()

    assert llm_client.generate_text("hi") == "fine"
    assert llm_client.usage_since(before)["total_tokens"] == 0


def test_failed_calls_add_nothing_to_the_count(api_mode, monkeypatch):
    capture(monkeypatch, FakeResponse(groq_error("boom"), status=500))

    before = llm_client.usage_snapshot()

    assert llm_client.generate_text("hi") is None
    assert llm_client.usage_since(before)["calls"] == 0


def test_each_model_has_its_own_pacing_budget(monkeypatch):
    class Clock:
        now, sleeps = 1000.0, []

        def monotonic(self):
            return self.now

        def sleep(self, s):
            self.sleeps.append(s)
            self.now += s

    clock = Clock()
    monkeypatch.setattr(llm_client.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(llm_client.time, "sleep", clock.sleep)
    monkeypatch.setattr(config, "API_TPM_LIMIT", 3000)
    llm_client.reset_pacing()

    for _ in range(3):
        llm_client._wait_for_budget(1000, "writer-model")          # the writer fills ITS minute

    llm_client._wait_for_budget(1000, "reviewer-model")            # the reviewer is not held up by that

    assert clock.sleeps == []

    llm_client._wait_for_budget(1000, "writer-model")              # but the writer itself must wait

    assert clock.sleeps
