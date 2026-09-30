from ai import llm_client
from database.models import Player
from game import feedback


def played(*results):
    """results: (theme, correct) tuples"""
    p = Player(name="t")

    for theme, ok in results:
        p.games_played += 1
        p.correct_answers += int(ok)
        feedback.record_round(p, theme, ok)

    return p


def test_summary_finds_strongest_and_weakest():
    p = played(("Space", True), ("Space", True), ("Finance", False), ("Finance", True), ("History", False))

    s = feedback.build_summary(p, 2)

    assert s["strongest"]["theme"] == "Space"
    assert s["weakest"]["theme"] == "History"
    assert s["accuracy"] == 60 and s["difficulty"] == "Medium"


def test_no_weakest_when_all_themes_equal():
    p = played(("Space", True), ("Finance", True))

    assert feedback.build_summary(p, 1)["weakest"] is None


def test_empty_session_message():
    s = feedback.build_summary(Player(name="t"), 1)

    assert "Play a few rounds" in feedback.rule_based_feedback(s)


def test_rule_based_feedback_mentions_real_numbers():
    s = feedback.build_summary(played(("Space", True), ("Finance", False)), 1)

    text = feedback.rule_based_feedback(s)

    assert "1 of 2" in text and "Space" in text and "Finance" in text


def test_ai_feedback_used_when_good_and_fallback_when_not(monkeypatch):
    s = feedback.build_summary(played(("Space", True), ("Finance", False)), 1)

    monkeypatch.setattr(llm_client, "is_available", lambda: True)

    monkeypatch.setattr(llm_client, "generate_text",
                        lambda *a, **k: "Nice work on Space! Try a few more Finance rounds to level up.")
    text, from_ai = feedback.get_session_feedback(s)
    assert from_ai is True and "Space" in text

    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: None)
    assert feedback.get_session_feedback(s)[1] is False

    monkeypatch.setattr(llm_client, "generate_text", lambda *a, **k: "ok")  # too short
    assert feedback.get_session_feedback(s)[1] is False


def test_feedback_greets_a_named_player_but_not_guests_or_defaults():
    def text(name):
        p = played(("Space", True), ("Finance", True))
        p.name = name

        return feedback.rule_based_feedback(feedback.build_summary(p, 1))

    assert text("Shreya").startswith("Excellent work, Shreya!")
    assert text("Guest").startswith("Excellent work!")
    assert text("Player").startswith("Excellent work!")
