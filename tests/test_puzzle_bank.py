import json

import config
from ai.puzzle_bank import load_bank, load_entries, pick_puzzle
from ai.validator import validate_wordle


def test_bank_covers_every_sector_and_difficulty():
    bank = load_bank()

    for theme in config.SECTORS:
        for level in config.DIFFICULTY_LEVELS:
            assert any(p.theme == theme and p.difficulty == level for p in bank), (theme, level)


def test_every_bank_entry_is_valid():
    seen = set()

    for e in load_entries():
        word = e["word"]

        assert validate_wordle(word)[0], word
        assert word not in seen, f"duplicate {word}"
        seen.add(word)

        assert word.lower() not in e["hint"].lower(), word
        assert word.lower() not in e["sentence"].lower(), word
        assert "_____" in e["sentence"], word
        assert len(e["options"]) == 4 and len(set(e["options"])) == 4, word
        assert e["options"][0] == word, word


def test_pick_avoids_used_words_and_matches_filters():
    p = pick_puzzle("Space", "Easy", used_words=["ROCKET"])

    assert p.theme == "Space" and p.difficulty == "Easy"
    assert p.word != "ROCKET"


def test_pick_allows_repeat_when_everything_used():
    words = [p.word for p in load_bank() if p.theme == "Space"]

    assert pick_puzzle("Space", "Easy", used_words=words) is not None


def test_pick_unknown_theme_still_returns_something():
    assert pick_puzzle("Not A Theme", "Easy") is not None


def test_missing_or_broken_file_returns_none(tmp_path):
    assert pick_puzzle("Space", "Easy", path=tmp_path / "nope.json") is None

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")

    assert pick_puzzle("Space", "Easy", path=bad) is None


def test_malformed_rows_are_skipped(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps([{"word": "ROCKET"}, {
        "theme": "Space", "difficulty": "Easy", "word": "ROCKET",
        "sentence": "_____", "options": ["ROCKET", "A", "B", "C"]}]))

    assert len(load_bank(f)) == 1
