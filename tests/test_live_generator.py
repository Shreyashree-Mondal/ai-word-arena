import json

import pytest

import config
from ai import live_generator
from ai.puzzle_bank import load_entries


def entry(word, **over):
    base = {
        "word": word,
        "hint": f"Clue for {word[::-1].lower()}.",
        "sentence": "The thing is called a _____ here.",
        "distractors": ["FOLDER", "CURSOR", "BUFFER"],
        "explanation": "Because it fits.",
    }
    base.update(over)
    return base


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.timeouts = []

    def __call__(self, prompt, **kwargs):
        self.calls += 1
        self.timeouts.append(kwargs.get("timeout"))
        return self.replies.pop(0) if self.replies else None


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    live_generator.clear_buffer()
    monkeypatch.setattr(config, "AUTO_SAVE_AI_PUZZLES", True)
    yield
    live_generator.clear_buffer()


def test_returns_valid_puzzle_and_buffers_the_spares(tmp_path):
    path = tmp_path / "bank.json"
    llm = FakeLLM(json.dumps([entry("SOCKET"), entry("KERNEL"), entry("BINARY")]))

    first = live_generator.generate_live_puzzle("Technology", "Medium", llm=llm, bank_path=path)

    assert first.word == "SOCKET" and first.theme == "Technology"
    assert llm.calls == 1 and llm.timeouts == [config.LIVE_AI_TIMEOUT]

    # next two rounds are served from the buffer: NO new AI call
    second = live_generator.generate_live_puzzle("Technology", "Medium", llm=llm, bank_path=path)
    third = live_generator.generate_live_puzzle("Technology", "Medium", llm=llm, bank_path=path)

    assert {second.word, third.word} == {"KERNEL", "BINARY"}
    assert llm.calls == 1


def test_buffered_puzzle_already_used_is_skipped(tmp_path):
    llm = FakeLLM(
        json.dumps([entry("SOCKET"), entry("KERNEL")]),
        json.dumps([entry("BINARY")]),
    )

    live_generator.generate_live_puzzle("Technology", "Medium", llm=llm, bank_path=tmp_path / "b.json")

    p = live_generator.generate_live_puzzle(
        "Technology", "Medium", used_words=["KERNEL"], llm=llm, bank_path=tmp_path / "b.json"
    )

    assert p.word == "BINARY"  # KERNEL was buffered but already played


def test_ai_down_returns_none_so_the_bank_can_take_over(tmp_path):
    llm = FakeLLM()  # always returns None

    assert live_generator.generate_live_puzzle(
        "Technology", "Easy", llm=llm, bank_path=tmp_path / "b.json"
    ) is None
    assert llm.calls == 1  # no pointless retries when the provider is down


def test_garbage_is_retried_then_gives_up(tmp_path):
    llm = FakeLLM("not json at all", "still nothing useful")

    assert live_generator.generate_live_puzzle(
        "Technology", "Easy", llm=llm, bank_path=tmp_path / "b.json"
    ) is None
    assert llm.calls == config.LIVE_MAX_ATTEMPTS


def test_invalid_entries_never_reach_the_player(tmp_path):
    bad = [
        entry("QQQQQQ"),                               # not a word
        entry("SOCKET", hint="A socket is a thing."),  # leaks the answer
        entry("KERNEL", distractors=["A", "B"]),       # wrong distractor count
        entry("BINARY"),                               # the only good one
    ]

    p = live_generator.generate_live_puzzle(
        "Technology", "Easy", llm=FakeLLM(json.dumps(bad)), bank_path=tmp_path / "b.json"
    )

    assert p.word == "BINARY"


def test_valid_ai_puzzles_are_saved_to_the_bank_without_duplicates(tmp_path):
    path = tmp_path / "bank.json"

    live_generator.generate_live_puzzle(
        "Technology", "Easy", llm=FakeLLM(json.dumps([entry("SOCKET"), entry("KERNEL")])), bank_path=path
    )

    assert {e["word"] for e in load_entries(path)} == {"SOCKET", "KERNEL"}

    live_generator.clear_buffer()
    live_generator.generate_live_puzzle(
        "Technology", "Easy", llm=FakeLLM(json.dumps([entry("SOCKET")])), bank_path=path
    )

    assert len(load_entries(path)) == 2  # SOCKET was not saved twice


def test_autosave_can_be_switched_off(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AUTO_SAVE_AI_PUZZLES", False)
    path = tmp_path / "bank.json"

    live_generator.generate_live_puzzle(
        "Technology", "Easy", llm=FakeLLM(json.dumps([entry("SOCKET")])), bank_path=path
    )

    assert not path.exists()
