"""Player names: safe, short, and never empty."""

import unicodedata

MAX_NAME_LEN = 20
GUEST_NAME = "Guest"

_EXTRA_ALLOWED = " -'._"


def _is_letter_or_digit(ch):
    return unicodedata.category(ch)[0] == "L" or unicodedata.category(ch) == "Nd"


def _allowed(ch):
    # L = letters, M = combining marks (vowel signs in Hindi, Tamil, ... need these),
    # Nd = digits. Everything else (symbols, markup, control characters) is dropped.
    return (
        unicodedata.category(ch)[0] in ("L", "M")
        or unicodedata.category(ch) == "Nd"
        or ch in _EXTRA_ALLOWED
    )


def clean_name(text):
    """
    Returns a tidy display name, or "" if nothing usable was typed.
    Keeps letters (any language), digits, spaces and - ' . _ only, so a
    name can never inject markup or break a prompt.
    """
    kept = "".join(ch for ch in (text or "") if _allowed(ch))

    kept = " ".join(kept.split()).strip(_EXTRA_ALLOWED)[:MAX_NAME_LEN].strip(_EXTRA_ALLOWED)

    return kept if any(_is_letter_or_digit(ch) for ch in kept) else ""


def is_named(name):
    """True for a real player name (not the default or Guest)."""
    return bool(name) and name not in ("Player", GUEST_NAME)


# ---------------------------------------
# PIN rules
# ---------------------------------------
import config  # noqa: E402


def clean_pin(text):
    """The PIN if it is 4-8 digits, otherwise "". Never strips or 'fixes' digits."""
    pin = text or ""

    if pin.isascii() and pin.isdigit() and config.PIN_MIN_LEN <= len(pin) <= config.PIN_MAX_LEN:
        return pin

    return ""


def pin_problem(pin):
    """
    Why a NEW PIN is not acceptable ("" if it is fine). Used when a PIN is
    created or changed, not at login.
    """
    if not clean_pin(pin):
        return f"The PIN must be {config.PIN_MIN_LEN} to {config.PIN_MAX_LEN} digits."

    if len(set(pin)) == 1:
        return "That PIN is too easy to guess (all the same digit)."

    steps = {int(b) - int(a) for a, b in zip(pin, pin[1:])}

    if steps in ({1}, {-1}):
        return "That PIN is too easy to guess (a straight run of digits)."

    return ""
