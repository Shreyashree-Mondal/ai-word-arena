from ai import hints, llm_client
from database.models import Puzzle

PUZZLE = Puzzle(
    word="ROCKET", sentence="_____", theme="Space", difficulty="Easy",
    hint="A vehicle that reaches orbit.",
)


def test_fallback_hints_never_contain_the_full_word_and_get_stronger():
    texts = [hints.fallback_hint(PUZZLE, n) for n in (1, 2, 3)]

    assert all("ROCKET" not in t.upper() for t in texts)
    assert len(set(texts)) == 3


def test_no_ai_gives_fallback(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: False)

    text, from_ai = hints.get_extra_hint(PUZZLE, [])

    assert from_ai is False and "6 letters" in text


def test_good_ai_hint_is_used(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "generate_text",
                        lambda *a, **k: 'Hint: "Astronauts ride this to leave Earth."')

    text, from_ai = hints.get_extra_hint(PUZZLE, [])

    assert from_ai is True
    assert text == "Astronauts ride this to leave Earth."


def test_ai_hint_that_leaks_the_answer_is_rejected(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "generate_text",
                        lambda *a, **k: "It is a rocket that flies to space.")

    text, from_ai = hints.get_extra_hint(PUZZLE, [])

    assert from_ai is False
    assert "rocket" not in text.lower()


def test_ai_failure_or_repeat_falls_back(monkeypatch):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)

    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: None)
    assert hints.get_extra_hint(PUZZLE, [])[1] is False

    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: "Same clue.")
    assert hints.get_extra_hint(PUZZLE, ["same clue."])[1] is False
