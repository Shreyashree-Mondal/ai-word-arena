import pytest

from game.players import GUEST_NAME, MAX_NAME_LEN, clean_name, is_named


@pytest.mark.parametrize("typed,expected", [
    ("Shreya", "Shreya"),
    ("  Shreya   Mondal  ", "Shreya Mondal"),
    ("O'Brien-Smith", "O'Brien-Smith"),
    ("Ravi_99", "Ravi_99"),
    ("Zoë", "Zoë"),
    ("श्रेया", "श्रेया"),
    ("ஸ்ரேயா", "ஸ்ரேயா"),
    ("শ্রেয়া", "শ্রেয়া"),
    ("José", "José"),
])
def test_normal_names_are_kept(typed, expected):
    assert clean_name(typed) == expected


@pytest.mark.parametrize("junk", [None, "", "   ", "!!!", "<>", "---", "...", "'"])
def test_unusable_names_become_empty(junk):
    assert clean_name(junk) == ""


def test_markup_and_symbols_are_stripped():
    out = clean_name('<script>alert("x")</script> **bold** [link](x)')

    assert not any(ch in out for ch in '<>"*[]()/\\')


def test_length_is_capped_and_never_ends_in_punctuation():
    out = clean_name("a" * 100)

    assert len(out) == MAX_NAME_LEN
    assert clean_name("Sam" + "-" * 30) == "Sam"


def test_is_named():
    assert is_named("Shreya")
    assert not is_named("Player") and not is_named(GUEST_NAME) and not is_named("")


# ---------------- PIN rules ----------------

from game.players import clean_pin, pin_problem  # noqa: E402


@pytest.mark.parametrize("pin", ["4827", "13579", "84293716", "0482", "9351"])
def test_valid_pins_are_accepted_as_typed(pin):
    assert clean_pin(pin) == pin and pin_problem(pin) == ""


@pytest.mark.parametrize("pin", [None, "", "123", "123456789", "12a4", "48 27", " 4827", "4827\n",
                                 "-482", "48.2", "٣٤٥٦", "４８２７"])
def test_wrong_format_pins_are_rejected(pin):
    assert clean_pin(pin) == ""
    assert "4 to 8 digits" in pin_problem(pin or "")


@pytest.mark.parametrize("pin", ["0000", "1111", "99999999", "1234", "4321", "12345678", "87654321", "3456"])
def test_obviously_guessable_pins_are_refused_when_creating_one(pin):
    assert clean_pin(pin) == pin                             # fine as a format...
    assert "too easy" in pin_problem(pin)                    # ...but not as a new PIN
