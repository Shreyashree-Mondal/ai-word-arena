import pytest

from agent import tools
from ai.puzzle_bank import load_entries
from nlp import lexicon

GOOD = {
    "word": "DRIVER",
    "hint": "The person who steers a racing car around the track.",
    "sentence": "The _____ took the final corner at full speed.",
    "distractors": ["MECHANIC", "COMMENTATOR", "SPECTATOR"],
    "explanation": "A driver controls the car.",
}


def box(difficulty="Easy", used=(), bank=()):
    return tools.ToolBox("Formula 1", difficulty, used, bank)


def test_check_word_accepts_a_good_word_and_reports_its_tier():
    r = box().check_word({"word": "driver"})

    assert r["valid_answer"] is True and r["tier"] == "Easy" and r["already_used"] is False
    assert r["definition"] and r["part_of_speech"]


@pytest.mark.parametrize("word,why", [
    ("QQQQQQ", "Not a valid English word"),
    ("PLANTS", "base form"),
    ("CAT", "exactly 6"),
    ("DRIV3R", "alphabetic"),
])
def test_check_word_explains_rejections(word, why):
    r = box().check_word({"word": word})

    assert r["valid_answer"] is False and why in r["reason"]


def test_used_and_banked_words_and_their_inflections_are_flagged():
    assert box(used=["DRIVERS"]).check_word({"word": "DRIVER"})["already_used"] is True
    assert box(bank=["DRIVER"]).check_duplicate({"word": "drivers"})["duplicate"] is True
    assert box().check_duplicate({"word": "GARAGE"})["duplicate"] is False


def test_check_leak():
    assert box().check_leak({"text": "Many drivers race.", "word": "DRIVER"})["leaks"] is True
    assert box().check_leak({"text": "Someone who steers.", "word": "DRIVER"})["leaks"] is False


def test_estimate_difficulty_and_acceptability():
    easy = box("Easy").estimate_difficulty({"word": "DRIVER"})
    hard_for_easy = box("Easy").estimate_difficulty({"word": "PISTON"})

    assert easy["tier"] == "Easy" and easy["acceptable"] is True
    assert hard_for_easy["tier"] == "Hard" and hard_for_easy["acceptable"] is False
    assert box("Medium").estimate_difficulty({"word": "PISTON"})["acceptable"] is True
    assert "error" in box().estimate_difficulty({"word": "QQQQQQ"})


def test_check_distractors_finds_every_problem():
    ok = box().check_distractors({"word": "DRIVER", "distractors": ["MECHANIC", "SPECTATOR", "PODIUM"]})

    assert ok["ok"] is True

    bad = box().check_distractors({"word": "DRIVER", "distractors": ["ZZZZZ", "DRIVERS", "PODIUM", "PODIUM"]})

    joined = " | ".join(bad["problems"])

    assert not bad["ok"]
    assert "ZZZZZ is not a real word" in joined
    assert "DRIVERS is a form of the answer" in joined
    assert "PODIUM is repeated" in joined
    assert "exactly 3" in joined

    assert "error" in box().check_distractors({"word": "DRIVER", "distractors": "A,B,C"})


def test_lookup_definition():
    assert box().lookup_definition({"word": "driver"})["found"] is True
    assert box().lookup_definition({"word": "qqqqqq"})["found"] is False


def test_call_never_raises_and_rejects_unknown_tools():
    tb = box()

    assert "Unknown tool" in tb.call("delete_everything", {})["error"]
    assert "Unknown tool" in tb.call("submit_puzzle", GOOD)["error"]   # submit is not a plain tool
    assert tb.call("check_word", "not a dict")["valid_answer"] is False
    assert tb.call("check_word", {"word": 12345})["valid_answer"] is False
    assert tb.call("check_distractors", {"word": None, "distractors": [None, 3, {}]})["ok"] is False


class TestSubmissionGate:
    def test_valid_puzzle_is_accepted(self):
        verdict, entry = box().validate_submission(GOOD)

        assert verdict == {"accepted": True, "problems": []}
        assert entry["word"] == "DRIVER" and entry["options"][0] == "DRIVER"

    @pytest.mark.parametrize("change,problem", [
        ({"word": "QQQQQQ"}, "Not a valid English word"),
        ({"hint": "A driver steers the car."}, "hint contains the answer"),
        ({"sentence": "No blank in here."}, "sentence has no blank"),
        ({"sentence": "The driver took the _____."}, "sentence contains the answer"),
        ({"distractors": ["MECHANIC", "SPECTATOR"]}, "need exactly 3 distractors"),
        ({"distractors": ["MECHANIC", "SPECTATOR", "ZXQWVB"]}, "distractor is not a real word"),
        ({"distractors": ["MECHANIC", "SPECTATOR", "DRIVERS"]}, "bad distractors"),
        ({"explanation": ""}, "bad explanation"),
    ])
    def test_every_rule_is_enforced(self, change, problem):
        verdict, entry = box().validate_submission({**GOOD, **change})

        assert entry is None and verdict["accepted"] is False
        assert problem in verdict["problems"][0]

    def test_duplicates_are_rejected(self):
        assert box(used=["DRIVERS"]).validate_submission(GOOD)[1] is None
        assert box(bank=["DRIVER"]).validate_submission(GOOD)[1] is None

    def test_word_two_levels_from_the_target_is_rejected(self):
        piston = {
            "word": "PISTON", "hint": "A part that slides inside an engine cylinder.",
            "sentence": "The _____ moved up and down inside the engine.",
            "distractors": ["GEARBOX", "RADIATOR", "CLUTCH"], "explanation": "Pistons move in cylinders.",
        }

        verdict, entry = box("Easy").validate_submission(piston)

        assert entry is None and "Hard-level" in verdict["problems"][0]
        assert box("Medium").validate_submission(piston)[1] is not None   # one level away is fine

    @pytest.mark.parametrize("junk", [None, "text", 5, [], {"word": None}, {"distractors": 7}])
    def test_garbage_never_crashes_the_gate(self, junk):
        verdict, entry = box().validate_submission(junk)

        assert entry is None and verdict["accepted"] is False


def test_frequency_bands_agree_with_the_hand_labelled_bank():
    entries = load_entries()

    exact = sum(tools.tier_for(e["word"]) == e["difficulty"] for e in entries)
    close = sum(tools.tier_distance(e["word"], e["difficulty"]) <= 1 for e in entries)

    assert exact / len(entries) >= 0.65      # measured: 71%
    assert close / len(entries) >= 0.90      # measured: 95%


# ---------------- check_words: many candidates in one call ----------------

def test_check_words_sorts_candidates_into_usable_and_rejected_with_reasons():
    result = box("Medium", used=["WICKET"]).check_words(
        {"words": ["batter", "wicket", "legbye", "bowled", "umpire", "stump"]}
    )

    assert [w["word"] for w in result["usable"]] == ["BATTER", "UMPIRE"]
    assert all(set(w) == {"word", "tier", "fits_difficulty"} for w in result["usable"])

    why = {r["word"]: r["why"] for r in result["rejected"]}

    assert why["WICKET"] == "already used"
    assert "Not a valid English word" in why["LEGBYE"]
    assert "base form" in why["BOWLED"]
    assert "exactly 6 letters" in why["STUMP"]


def test_check_words_flags_which_candidates_suit_the_target_difficulty():
    result = box("Easy").check_words({"words": ["driver", "piston"]})

    fits = {w["word"]: w["fits_difficulty"] for w in result["usable"]}

    assert fits == {"DRIVER": True, "PISTON": False}       # PISTON is a Hard-tier word


def test_check_words_ignores_repeats_and_blanks_and_caps_the_batch():
    result = box().check_words({"words": ["driver", "DRIVER", "", "  ", "garage"] + ["qqqqqq"] * 30})

    assert [w["word"] for w in result["usable"]] == ["DRIVER", "GARAGE"]
    assert len(result["usable"]) + len(result["rejected"]) <= tools.MAX_BATCH


def test_check_words_accepts_a_comma_separated_string_too():
    result = box().check_words({"words": "driver, garage  helmet"})

    assert [w["word"] for w in result["usable"]] == ["DRIVER", "GARAGE", "HELMET"]


@pytest.mark.parametrize("bad", [None, 5, {"a": 1}, []])
def test_check_words_survives_bad_arguments(bad):
    result = box().check_words({"words": bad})

    assert "error" in result or result == {"usable": [], "rejected": []}


def test_check_words_is_a_listed_tool_and_reachable_through_call():
    assert "check_words" in tools.TOOL_DOCS and list(tools.TOOL_DOCS)[0] == "check_words"
    assert box().call("check_words", {"words": ["driver"]})["usable"][0]["word"] == "DRIVER"
