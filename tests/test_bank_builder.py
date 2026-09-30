import json

from ai import bank_builder
from ai.puzzle_bank import load_entries

GOOD = {
    "word": "SOCKET", "hint": "An endpoint for two programs to talk.",
    "sentence": "The app opened a network ___ to the server.",
    "distractors": ["FOLDER", "CURSOR", "BUFFER"], "explanation": "A socket is a network endpoint.",
}


def check(raw, taken=()):
    return bank_builder.validate_entry(raw, "Technology", "Medium", set(taken))


def test_valid_entry_is_normalised():
    entry, reason = check(GOOD)

    assert reason == "ok"
    assert entry["options"][0] == "SOCKET" and len(entry["options"]) == 4
    assert entry["sentence"].count("_____") == 1  # ___ normalised to _____


def test_rejections():
    assert check({**GOOD, "word": "QQQQQQ"})[0] is None            # not a word
    assert check({**GOOD, "word": "CAT"})[0] is None                # wrong length
    assert check(GOOD, taken={"SOCKET"})[1] == "duplicate word"
    assert check({**GOOD, "hint": "A socket is this."})[1] == "hint contains the answer"
    assert check({**GOOD, "sentence": "No blank here."})[1] == "sentence has no blank"
    assert check({**GOOD, "distractors": ["A", "B"]})[1] == "need exactly 3 distractors"
    assert check({**GOOD, "distractors": ["SOCKET", "B", "C"]})[1] == "bad distractors"
    assert check({"word": "SOCKET"})[1] == "missing fields"


def test_parse_handles_fences_chatter_and_garbage():
    fenced = "Sure!\n```json\n" + json.dumps([GOOD]) + "\n```\nHope that helps."

    assert bank_builder.parse_entries(fenced) == [GOOD]
    assert bank_builder.parse_entries(json.dumps(GOOD)) == [GOOD]
    assert bank_builder.parse_entries("nothing useful") == []
    assert bank_builder.parse_entries(None) == []


def test_build_bank_tops_up_and_saves(tmp_path):
    path = tmp_path / "bank.json"

    words = iter(["SOCKET", "KERNEL", "BINARY", "CIPHER"])

    def fake_llm(prompt, **kwargs):
        w = next(words)
        return json.dumps([{**GOOD, "word": w, "hint": f"Clue about {w[::-1].lower()}.",
                            "sentence": "Something ___ here."}])

    stats = bank_builder.build_bank(
        ["Technology"], ["Medium"], target=2, path=path,
        llm=fake_llm, batch_size=1, log=lambda *_: None,
    )

    assert stats["added"] == 2
    assert len(load_entries(path)) == 2


def test_build_bank_stops_cleanly_when_ai_is_unavailable(tmp_path):
    stats = bank_builder.build_bank(
        ["Technology"], ["Easy"], target=3, path=tmp_path / "b.json",
        llm=lambda *a, **k: None, log=lambda *_: None,
    )

    assert stats["aborted"] is True and stats["added"] == 0


def test_inflection_of_a_taken_word_is_a_duplicate():
    assert check(GOOD, taken={"SOCKETS"})[1] == "duplicate word"


def test_hint_leaking_an_inflection_or_compound_is_rejected():
    assert check({**GOOD, "hint": "Many sockets exist."})[1] == "hint contains the answer"
    assert check({**GOOD, "hint": "A socketed thing."})[1] == "hint contains the answer"


def test_made_up_distractors_are_rejected():
    assert check({**GOOD, "distractors": ["FOLDER", "CURSOR", "ZXQWVB"]})[1] == "distractor is not a real word"


def test_distractor_that_is_a_form_of_the_answer_is_rejected():
    assert check({**GOOD, "distractors": ["SOCKETS", "CURSOR", "BUFFER"]})[1] == "bad distractors"
