import json

import pytest

import config
from agent.puzzle_agent import PuzzleAgent, parse_action
from agent_fakes import GARAGE, GOOD, ScriptedLLM, act, submit

def run(llm, difficulty="Easy", used=(), bank=(), critic=False, **kw):
    agent = PuzzleAgent(llm=llm, use_critic=critic, **kw)

    return agent.run("Formula 1", difficulty, used, bank)


# ---------------- the happy path ----------------

def test_agent_uses_tools_then_submits_a_valid_puzzle():
    llm = ScriptedLLM(
        act("check_word", "first candidate", word="DRIVER"),
        act("check_leak", "make sure the hint is clean", text=GOOD["hint"], word="DRIVER"),
        submit(),
    )

    result = run(llm)

    assert result.success and result.reason == "puzzle accepted"
    assert result.entry["word"] == "DRIVER" and result.entry["theme"] == "Formula 1"
    assert [s.action for s in result.steps] == ["check_word", "check_leak", "submit_puzzle"]
    assert result.llm_calls == 3
    assert result.steps[0].observation["valid_answer"] is True
    assert result.steps[1].observation == {"leaks": False}


def test_prompt_carries_theme_rules_tools_used_words_and_history():
    llm = ScriptedLLM(act("check_word", word="DRIVER"), submit())

    run(llm, used=["GARAGE"])

    first, second = llm.prompts

    for needle in ["Formula 1", "Easy", "check_word", "submit_puzzle", "GARAGE", "EXACTLY ONE JSON"]:
        assert needle in first

    assert "Observation 1" in second and "valid_answer" in second   # tool result fed back


# ---------------- recovering from mistakes ----------------

def test_recovers_from_an_invalid_word_using_the_tool_result():
    llm = ScriptedLLM(
        act("check_word", "try this", word="QQQQQQ"),
        act("check_word", "that was invalid, try another", word="GARAGE"),
        submit(GARAGE),
    )

    result = run(llm)

    assert result.steps[0].observation["valid_answer"] is False
    assert result.success and result.entry["word"] == "GARAGE"


def test_leaky_submission_is_rejected_with_the_reason_then_fixed():
    llm = ScriptedLLM(
        submit(hint="A driver steers the car."),
        submit(),
    )

    result = run(llm)

    assert result.steps[0].ok is False
    assert result.steps[0].problems == ["hint contains the answer"]
    assert result.success and result.steps[1].ok is True
    assert "hint contains the answer" in llm.prompts[1]   # the agent was told what to fix


@pytest.mark.parametrize("kind", ["used", "bank"])
def test_reused_words_are_rejected_even_as_inflections(kind):
    taken = {kind if kind == "used" else "bank": ["DRIVERS"]}

    llm = ScriptedLLM(submit(), submit(GARAGE))

    result = run(llm, **({"used": ["DRIVERS"]} if kind == "used" else {"bank": ["DRIVERS"]}))

    assert result.steps[0].problems == ["duplicate word"]
    assert result.entry["word"] == "GARAGE"
    assert taken  # (keeps the parametrised dict referenced)


def test_word_far_from_target_difficulty_is_rejected():
    piston = dict(
        word="PISTON", hint="A part that slides inside an engine cylinder.",
        sentence="The _____ moved up and down inside the engine.",
        distractors=["GEARBOX", "RADIATOR", "CLUTCH"], explanation="Pistons move in cylinders.",
    )

    result = run(ScriptedLLM(submit(piston), submit()), difficulty="Easy")

    assert "Hard-level" in result.steps[0].problems[0]
    assert result.entry["word"] == "DRIVER"


def test_broken_json_and_unknown_tools_cost_a_step_but_nothing_else():
    llm = ScriptedLLM(
        "I think we should pick DRIVER!",
        act("hack_the_planet"),
        act("check_word", word="DRIVER"),
        submit(),
    )

    result = run(llm)

    assert result.steps[0].action == "invalid_reply" and result.steps[0].ok is False
    assert "Unknown tool" in result.steps[1].observation["error"]
    assert result.success and len(result.steps) == 4


def test_submission_written_without_an_args_wrapper_is_still_understood():
    flat = json.dumps({"thought": "done", "action": "submit_puzzle", **GOOD})

    assert run(ScriptedLLM(flat)).success


# ---------------- limits and failure ----------------

def test_step_budget_ends_the_run_without_a_puzzle():
    llm = ScriptedLLM(*[act("check_word", word="DRIVER")] * 10)

    result = run(llm, max_steps=3)

    assert not result.success and result.entry is None
    assert result.reason == "step budget exhausted" and result.llm_calls == 3


def test_unavailable_llm_stops_immediately():
    result = run(ScriptedLLM())

    assert result.reason == "LLM unavailable" and result.llm_calls == 1 and result.steps == []


def test_time_budget_is_enforced():
    ticks = iter(range(0, 10_000, 30))   # every clock read = 30 seconds later

    llm = ScriptedLLM(*[act("check_word", word="DRIVER")] * 10)

    result = run(llm, time_budget=45, clock=lambda: next(ticks))

    assert result.reason == "time budget exhausted" and result.llm_calls == 1


# ---------------- the agent cannot bypass validation ----------------

def test_agent_cannot_get_an_invalid_puzzle_accepted_by_any_means():
    llm = ScriptedLLM(
        act("submit_puzzle", word="QQQQQQ", hint="h", sentence="_____", distractors=["A", "B", "C"], explanation="e"),
        act("finish", "I am sure it is fine", puzzle="whatever"),
        act("accept_puzzle", word="QQQQQQ"),
        submit(word=123, distractors="MECHANIC,SPECTATOR,PODIUM"),
        act("check_word", "", ),                    # missing argument entirely
        {"action": "check_word", "args": "DRIVER"},  # args of the wrong type
    )

    result = run(llm, max_steps=6)

    assert result.entry is None and not result.success
    assert not any(s.ok and s.action == "submit_puzzle" for s in result.steps)


# ---------------- the critic ----------------

APPROVE = {"fits_theme": True, "ambiguous_options": [], "issue": ""}


def test_critic_rejection_sends_the_agent_back_to_fix_it():
    llm = ScriptedLLM(
        submit(),                                                          # agent
        {"fits_theme": True, "ambiguous_options": ["SPECTATOR"], "issue": ""},  # critic rejects
        submit(distractors=["MECHANIC", "COMMENTATOR", "SPONSOR"]),        # agent fixes
        APPROVE,                                                           # critic approves
    )

    result = run(llm, critic=True)

    assert result.success and result.llm_calls == 4
    assert "SPECTATOR could also fit" in result.steps[0].problems[0]
    assert "SPONSOR" in result.entry["options"]


def test_critic_can_reject_for_theme_fit():
    llm = ScriptedLLM(submit(), {"fits_theme": False, "ambiguous_options": []}, submit(GARAGE), APPROVE)

    result = run(llm, critic=True)

    assert "does not clearly fit the theme" in result.steps[0].problems[0]
    assert result.entry["word"] == "GARAGE"


@pytest.mark.parametrize("critic_reply", ["no idea what you mean", "{not json", None])
def test_unreadable_or_missing_critic_never_blocks_a_valid_puzzle(critic_reply):
    replies = [submit()] + ([critic_reply] if critic_reply is not None else [])

    assert run(ScriptedLLM(*replies), critic=True).success


def test_critic_naming_the_answer_or_a_stranger_is_ignored():
    llm = ScriptedLLM(submit(), {"fits_theme": True, "ambiguous_options": ["DRIVER", "BANANA"]})

    assert run(llm, critic=True).success


def test_a_puzzle_the_critic_keeps_rejecting_is_never_accepted():
    reject = {"fits_theme": True, "ambiguous_options": ["SPECTATOR"]}

    llm = ScriptedLLM(submit(), reject, submit(), reject, submit(), reject, submit(), APPROVE)

    result = run(llm, critic=True)

    assert config.AGENT_MAX_CRITIC_REJECTIONS == 2
    assert not result.success and result.entry is None
    assert result.reason == "critic kept rejecting the puzzle"
    assert result.llm_calls == 6                                   # 3 submissions + 3 reviews, then it stops
    assert all(not s.ok for s in result.steps)


def test_the_critic_may_send_the_agent_back_up_to_the_limit_and_then_approve():
    reject = {"fits_theme": True, "ambiguous_options": ["SPECTATOR"]}

    llm = ScriptedLLM(submit(), reject, submit(), reject, submit(), APPROVE)

    result = run(llm, critic=True)

    assert result.success and result.llm_calls == 6


def test_critic_is_never_asked_about_a_puzzle_the_validators_rejected():
    llm = ScriptedLLM(submit(hint="The driver steers."))

    result = run(llm, critic=True, max_steps=1)

    assert not result.success and result.llm_calls == 1 and len(llm.prompts) == 1


# ---------------- parsing ----------------

def test_parse_action_variants():
    assert parse_action('{"thought":"t","action":"check_word","args":{"word":"X"}}') == ("t", "check_word", {"word": "X"})
    assert parse_action('Sure!\n```json\n{"action":"check_word","args":{"word":"X"}}\n```') == ("", "check_word", {"word": "X"})
    assert parse_action('{"action":"a","args":{}} {"action":"b","args":{}}')[1] == "a"       # first object wins
    assert parse_action('{"action":"check_word","args":"oops"}') == ("", "check_word", {})   # bad args -> {}
    assert parse_action('{"action":"submit_puzzle","word":"X"}')[2] == {"word": "X"}          # flat submit


@pytest.mark.parametrize("bad", [None, "", "no json", "{broken", '{"thought":"x"}', '{"action": 5}', "[1,2]"])
def test_parse_action_rejects_junk(bad):
    assert parse_action(bad) is None


# ---------------- models that use their built-in tool-call format ----------------

def native(name, **arguments):
    return json.dumps({"name": name, "arguments": arguments})


def test_parse_action_understands_the_native_tool_call_format():
    assert parse_action('{"name": "check_word", "arguments": {"word":"helmet"}}') == ("", "check_word", {"word": "helmet"})
    assert parse_action('{"name":"check_word","arguments":"{\\"word\\": \\"helmet\\"}"}')[2] == {"word": "helmet"}
    assert parse_action('{"name":"submit_puzzle","arguments":{"word":"X"}}') == ("", "submit_puzzle", {"word": "X"})
    assert parse_action('{"name":"check_word","arguments":"not json"}') == ("", "check_word", {})
    assert parse_action('{"action":"check_word","name":"ignored","args":{"word":"X"}}')[1] == "check_word"   # ours wins


@pytest.mark.parametrize("bad", ['{"name": 5, "arguments": {}}', '{"name": "  "}', '{"arguments": {"word": "X"}}'])
def test_native_format_junk_is_still_rejected(bad):
    assert parse_action(bad) is None


def test_agent_completes_a_puzzle_speaking_only_the_native_format():
    llm = ScriptedLLM(
        native("check_word", word="DRIVER"),
        native("check_leak", text=GOOD["hint"], word="DRIVER"),
        json.dumps({"name": "submit_puzzle", "arguments": {k: v for k, v in GOOD.items()}}),
    )

    result = run(llm)

    assert result.success and result.entry["word"] == "DRIVER"
    assert [s.action for s in result.steps] == ["check_word", "check_leak", "submit_puzzle"]
    assert result.steps[0].observation["valid_answer"] is True


def test_native_format_submission_with_arguments_as_text_is_accepted_and_still_validated():
    good = json.dumps({"name": "submit_puzzle", "arguments": json.dumps(GOOD)})
    leaky = json.dumps({"name": "submit_puzzle", "arguments": json.dumps({**GOOD, "hint": "A driver steers."})})

    result = run(ScriptedLLM(leaky, good))

    assert result.steps[0].problems == ["hint contains the answer"]      # the gate still applies
    assert result.success


# ---------------- the critic: the real "called off due to rain" failure ----------------

RAIN = dict(
    word="HOCKEY",
    hint="A team sport played with sticks and a puck.",
    sentence="The _____ was called off due to rain.",
    distractors=["SOCCER", "TENNIS", "BASKET"],
    explanation="Hockey is a team sport.",
)


def test_vague_sentence_that_fits_several_options_is_sent_back_and_fixed():
    verdicts = [
        {"options": {"SOCCER": True, "TENNIS": True, "BASKET": False}, "fits_theme": True, "issue": ""},
        {"options": {"SOCCER": False, "TENNIS": False, "BASKET": False}, "fits_theme": True, "issue": ""},
    ]
    better = "Players use sticks to hit a puck across the ice in a game of _____."

    llm = ScriptedLLM(submit(RAIN), verdicts[0], submit(RAIN, sentence=better), verdicts[1])

    result = run(llm, critic=True)

    problem = result.steps[0].problems[0]

    assert "SOCCER, TENNIS could also fit" in problem and "BASKET" not in problem
    assert "more specific" in problem                                    # the agent is told HOW to fix it
    assert result.steps[0].observation["reviewer"] == config.CRITIC_MODEL
    assert result.success and result.entry["sentence"] == better


@pytest.mark.parametrize("verdict", [
    {"options": {"SOCCER": "yes", "TENNIS": "no", "BASKET": "no"}},
    {"options": {"soccer": True, "tennis": False, "basket": False}},                 # any letter case
    {"options": {"SOCCER": {"fits": True}, "TENNIS": {"fits": False}, "BASKET": {}}},
    {"options": {"SOCCER": False}, "ambiguous_options": ["SOCCER"]},                 # older format too
    {"ambiguous_options": ["soccer"]},
])
def test_the_reviewers_verdict_is_understood_in_every_common_spelling(verdict):
    result = run(ScriptedLLM(submit(RAIN), verdict), critic=True, max_steps=1)

    assert not result.success and "SOCCER could also fit" in result.steps[0].problems[0]


@pytest.mark.parametrize("verdict", [
    {"options": {"SOCCER": False, "TENNIS": False, "BASKET": False}, "fits_theme": True},
    {"options": {"SOCCER": "no", "TENNIS": "no"}},
    {"options": {"BANANA": True, "HOCKEY": True}},          # things that are not wrong options are ignored
    {"options": "nonsense"},
    {},
])
def test_an_approving_or_unhelpful_verdict_lets_a_valid_puzzle_through(verdict):
    assert run(ScriptedLLM(submit(RAIN), verdict), critic=True).success


def test_reviewer_uses_the_configured_model_and_the_writer_uses_the_default():
    llm = ScriptedLLM(submit(), APPROVE)

    run(llm, critic=True)

    assert llm.kwargs[0].get("model") is None                       # the agent's own call
    assert llm.kwargs[1].get("model") == config.CRITIC_MODEL        # the reviewer's call


def test_reviewer_falls_back_to_the_main_model_and_remembers_it_was_unavailable(monkeypatch):
    calls = []

    def llm(prompt, **kw):
        calls.append(kw.get("model"))

        if kw.get("model") == "big-reviewer":
            return None                                            # e.g. not available to this key

        return json.dumps(submit()) if "Your next JSON action" in prompt else json.dumps(APPROVE)

    monkeypatch.setattr(config, "CRITIC_MODEL", "big-reviewer")

    assert run(llm, critic=True).success
    assert calls == [None, "big-reviewer", None]                   # tried it, then fell back

    calls.clear()
    assert run(llm, critic=True).success
    assert calls == [None, None]                                   # did not try the broken model again


def test_an_empty_reviewer_setting_means_the_main_model_reviews(monkeypatch):
    monkeypatch.setattr(config, "CRITIC_MODEL", "")

    llm = ScriptedLLM(submit(), APPROVE)

    run(llm, critic=True)

    assert llm.kwargs[1].get("model") is None


# ---------------- tools named with a prefix ----------------

@pytest.mark.parametrize("given,expected", [
    ("tool.check_word", "check_word"), ("functions.check_word", "check_word"),
    ("Tool.Check_Word", "check_word"), ("functions/submit_puzzle", "submit_puzzle"),
    ("CHECK_WORD", "check_word"), ("a.b.check_leak", "check_leak"),
    ("hack_the_planet", "hack_the_planet"), ("tool.hack", "tool.hack"),
])
def test_prefixed_tool_names_are_understood_but_unknown_tools_stay_unknown(given, expected):
    from agent.puzzle_agent import normalise_tool_name

    assert normalise_tool_name(given) == expected


def test_agent_completes_a_puzzle_using_prefixed_tool_names():
    llm = ScriptedLLM(
        json.dumps({"name": "tool.check_word", "arguments": {"word": "DRIVER"}}),
        json.dumps({"name": "functions.submit_puzzle", "arguments": GOOD}),
    )

    result = run(llm)

    assert result.success and [s.action for s in result.steps] == ["check_word", "submit_puzzle"]


def test_a_used_word_is_reported_as_unusable_with_advice():
    from agent.tools import ToolBox

    box = ToolBox("Sports", "Easy", used_words=["TENNIS"], bank_words=["SOCCER"])

    used = box.check_word({"word": "tennis"})
    banked = box.check_word({"word": "SOCCER"})
    fresh = box.check_word({"word": "HOCKEY"})

    assert used["usable"] is False and "different word" in used["advice"]
    assert banked["usable"] is False and fresh["usable"] is True and "advice" not in fresh


# ---------------- a garbled model reply must not end the run ----------------

def test_an_empty_or_unparseable_reply_costs_one_step_and_the_run_carries_on():
    llm = ScriptedLLM("", act("check_word", word="DRIVER"), "<|garbled|>", submit())

    result = run(llm)

    assert result.success and result.reason == "puzzle accepted"
    assert [s.action for s in result.steps] == ["invalid_reply", "check_word", "invalid_reply", "submit_puzzle"]
    assert result.llm_calls == 4


def test_only_a_missing_reply_means_the_ai_is_unavailable():
    garbled_forever = ScriptedLLM(*[""] * (config.AGENT_MAX_STEPS + 2))

    assert run(garbled_forever).reason == "step budget exhausted"         # garbled replies: still "available"
    assert run(ScriptedLLM()).reason == "LLM unavailable"                 # nothing came back at all


@pytest.mark.parametrize("text,expected", [
    ('<|channel|>commentary to=functions.check_word <|constrain|>json<|message|>{"word":"wicket"}<|call|>',
     ("", "check_word", {"word": "wicket"})),
    ('to=tool.submit_puzzle json {"word":"WICKET","hint":"h"}', ("", "submit_puzzle", {"word": "WICKET", "hint": "h"})),
    ('junk to=functions.check_leak more junk {"text":"a","word":"b"}', ("", "check_leak", {"text": "a", "word": "b"})),
])
def test_a_models_built_in_call_syntax_is_understood(text, expected):
    assert parse_action(text) == expected


@pytest.mark.parametrize("text", [
    '<|channel|>commentary to=functions.hack_the_world<|message|>{"x": 1}',     # not one of our tools
    '{"word": "wicket"}',                                                        # arguments with no tool at all
    "to=functions.check_word",                                                   # a name with no arguments
])
def test_call_syntax_naming_no_known_tool_is_rejected(text):
    assert parse_action(text) is None


def test_the_real_failure_end_to_end_through_the_real_client(monkeypatch):
    """
    What a user hit: Groq answered 400 "Parsing failed" in the middle of a run.
    Before the fix the whole run was abandoned; now it recovers and finishes.
    """
    import config as cfg
    from ai import llm_client

    monkeypatch.setattr(cfg, "AI_PROVIDER", "api")
    monkeypatch.setattr(cfg, "API_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class Resp:
        def __init__(self, status, payload):
            self.status_code, self.ok, self._p, self.text = status, status < 400, payload, json.dumps(payload)

        def json(self):
            return self._p

    def ok(content):
        return Resp(200, {"choices": [{"message": {"content": content}}]})

    queue = [
        Resp(400, {"error": {"message": "Parsing failed. The model generated output that could not be parsed. "
                                        "Please adjust your prompt. See 'failed_generation' for more details."}}),
        ok(json.dumps(act("check_word", word="DRIVER"))),
        Resp(400, {"error": {"message": "x", "code": "tool_use_failed",
                             "failed_generation": json.dumps({"name": "submit_puzzle", "arguments": GOOD})}}),
    ]

    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: queue.pop(0))

    result = PuzzleAgent(use_critic=False).run("Formula 1", "Easy", [], [])

    assert result.success and result.entry["word"] == "DRIVER" and result.llm_calls == 3
    assert [s.action for s in result.steps] == ["invalid_reply", "check_word", "submit_puzzle"]


# ---------------- the critic's verdict is visible in the trace ----------------

def test_an_accepted_puzzle_shows_who_reviewed_it_and_what_they_said():
    result = run(ScriptedLLM(submit(), APPROVE), critic=True)

    verdict = result.steps[-1].observation

    assert verdict["accepted"] is True
    assert verdict["review"] == "critic approved" and verdict["reviewer"] == config.CRITIC_MODEL


def test_an_unavailable_critic_is_visible_rather_than_silent():
    result = run(ScriptedLLM(submit()), critic=True)          # the critic gets no reply

    assert result.success and result.steps[-1].observation["review"] == "critic unavailable"


def test_without_a_critic_the_verdict_carries_no_review_fields():
    verdict = run(ScriptedLLM(submit()), critic=False).steps[-1].observation

    assert verdict == {"accepted": True, "problems": []}


# ---------------- token economy: small prompts, small replies ----------------

def long_history(n):
    from agent.puzzle_agent import Step
    from agent.tools import ToolBox

    box = ToolBox("Cricket", "Medium", [], [])

    return [
        Step(i, "thinking hard about which word would make the best puzzle", "check_word",
             {"word": "wicket"}, box.check_word({"word": "wicket"}), True)
        for i in range(1, n + 1)
    ]


def build_prompt(steps):
    from agent.tools import ToolBox

    return PuzzleAgent(llm=lambda *a, **k: None)._prompt("Cricket", "Medium", ToolBox("Cricket", "Medium", [], []), steps)


def test_the_prompt_stops_growing_however_long_the_run():
    sizes = {n: len(build_prompt(long_history(n))) for n in (0, 3, 8, 30, 100)}

    assert sizes[0] < 2600 and sizes[100] < 4200                   # the old prompt passed 5,200 by step 7
    assert sizes[100] == sizes[30] + 0 or sizes[100] - sizes[30] < 30    # completely flat once the cap is reached
    assert sizes[100] - sizes[8] < 400


def test_only_the_latest_steps_are_shown_in_full_and_older_ones_in_one_line():
    prompt = build_prompt(long_history(8))

    assert "Earlier steps:" in prompt
    assert prompt.count("Observation ") == 3                       # steps 6, 7, 8 in full
    assert "1. check_word -> not usable: Already used" not in prompt   # (WICKET is not used in this box)
    assert "  1. check_word -> usable" in prompt and "  5. check_word -> usable" in prompt


def test_noisy_fields_are_dropped_from_the_history_but_the_useful_ones_stay():
    prompt = build_prompt(long_history(2))

    assert "zipf" not in prompt and "part_of_speech" not in prompt
    assert '"tier"' in prompt and '"usable"' in prompt and '"definition"' in prompt


def test_a_batch_check_is_summarised_in_one_line_when_it_becomes_history():
    from agent.puzzle_agent import Step
    from agent.tools import ToolBox

    box = ToolBox("Cricket", "Medium", ["WICKET"], [])
    batch = Step(1, "", "check_words", {}, box.check_words({"words": ["batter", "wicket", "legbye"]}), True)
    filler = long_history(3)

    for i, s in enumerate(filler, start=2):
        s.index = i

    prompt = PuzzleAgent(llm=lambda *a, **k: None)._prompt("Cricket", "Medium", box, [batch] + filler)

    line = next(l for l in prompt.splitlines() if l.strip().startswith("1. check_words"))

    assert "usable: BATTER" in line and "WICKET (already used)" in line and "LEGBYE" in line


def test_older_rejections_are_remembered_so_the_agent_does_not_repeat_itself():
    from agent.puzzle_agent import Step

    rejected = Step(1, "", "submit_puzzle", {}, {"accepted": False, "problems": ["hint contains the answer"]}, False)
    prompt = build_prompt([rejected] + long_history(4))

    assert "rejected: hint contains the answer" in prompt


def test_each_step_asks_for_only_a_small_reply(monkeypatch):
    llm = ScriptedLLM(submit())

    run(llm)

    assert llm.kwargs[0]["max_tokens"] == config.AGENT_MAX_TOKENS <= 400


def test_the_agent_is_pointed_at_the_batch_tool_first():
    assert "check_words" in build_prompt([]) and "ALL in one call" in build_prompt([])


def test_a_run_that_uses_the_batch_tool_finishes_in_three_calls():
    llm = ScriptedLLM(
        act("check_words", "brainstorm", words=["legbye", "bowled", "driver", "garage"]),
        act("check_leak", text=GOOD["hint"], word="DRIVER"),
        submit(),
    )

    result = run(llm)

    assert result.success and result.llm_calls == 3
    assert [w["word"] for w in result.steps[0].observation["usable"]] == ["DRIVER", "GARAGE"]


# ---------------- keeping the model away from words this theme already used ----------------

def used_line(prompt):
    return next(line for line in prompt.splitlines() if line.startswith("Already used"))


def test_the_prompt_lists_this_themes_words_first_then_session_words_and_stays_short():
    llm = ScriptedLLM(act("check_word", word="DRIVER"))
    theme_words = [f"THEME{i:02d}" for i in range(60)]

    PuzzleAgent(llm=llm, use_critic=False, max_steps=1).run(
        "Sports", "Easy", ["SESSION"], ["ROCKET", "PLANET"], theme_words)

    line = used_line(llm.prompts[0])
    listed = [w.strip() for w in line.split(":", 1)[1].split(",")]

    assert listed[0] == "THEME00" and len(listed) == 40                  # capped
    assert "ROCKET" not in line and "PLANET" not in line                  # other themes' words are left out


def test_session_words_are_listed_when_the_theme_has_few_words():
    llm = ScriptedLLM(act("check_word", word="DRIVER"))

    PuzzleAgent(llm=llm, use_critic=False, max_steps=1).run("Sports", "Easy", ["SESSION"], [], ["SOCCER"])

    assert used_line(llm.prompts[0]).endswith("SOCCER, SESSION")


def test_without_theme_words_the_old_behaviour_is_kept():
    llm = ScriptedLLM(act("check_word", word="DRIVER"))

    PuzzleAgent(llm=llm, use_critic=False, max_steps=1).run("Sports", "Easy", ["ZEBRAS"], ["APPLES"])

    line = used_line(llm.prompts[0])

    assert "ZEBRAS" in line and "APPLES" in line


# ---------------------------------------------------------------- tokens per run ----------------

def test_a_run_reports_the_tokens_the_provider_counted(monkeypatch):
    from ai import llm_client

    def metered(prompt, **kwargs):
        llm_client._record_usage({"prompt_tokens": 500, "completion_tokens": 40})
        return json.dumps(submit())

    result = PuzzleAgent(llm=metered, use_critic=False).run("Formula 1", "Easy", [], [])

    assert result.success and result.tokens == 540


def test_a_run_with_a_fake_llm_reports_zero_tokens():
    assert run(ScriptedLLM(submit())).tokens == 0
