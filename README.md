# AI Word Arena

A vocabulary game built with Streamlit. Pick a category, then play:

- **MCQ** - fill in the blank and choose the right word
- **Wordle** - guess the 6-letter word from a clue (green / yellow / gray feedback)
- **Anagram** - unscramble the letters (3 attempts). Any real word made from the letters
  counts, e.g. SILENT is accepted for LISTEN.
- **Star** (Spelling-Bee style) - 7 letters with one in the center. Find words of 4+ letters
  that use only those letters and always include the center letter. One or more **pangrams**
  use all 7 letters.
  - 4-letter word = 1 point, longer words = their length, pangram = +7 bonus
  - ranks from Beginner to Genius by % of the maximum points; 35% ("Good") counts as a win
  - hints, letter shuffle, and a "words you missed" list at the end
  - every solution word is computed from the dictionary, so the word list is real by
    construction. The AI may only suggest a *themed pangram*, and code verifies it
    (7 distinct letters, real word, big enough word list) before it is used; otherwise the
    puzzle is built straight from the dictionary. Plurals and verb forms count as separate
    words here (as in Spelling Bee); each word can only be scored once.

## Log in, PIN, saved progress

Choose **New player** to create a profile: a name and a 4 to 8 digit **PIN** (typed twice).
Come back with **Log in** and the same name and PIN to restore everything: score, streaks,
difficulty level, per-category results, hints used, the puzzles you have already seen, and your
recent rounds. Progress is saved automatically after every round. **Log out** is in the sidebar,
a **Top players** list shows the best scores, and **Play as guest** plays without saving anything.

How the PIN protects you:

- It is stored only as a **salted PBKDF2 hash**, never as digits, and is never kept in the session.
- After **5 wrong PINs a profile is locked for 5 minutes**, even against the correct PIN, so a
  short PIN cannot be guessed through the game. Wrong tries in Account settings count too.
- Names are unique (letter case ignored), so nobody can register over an existing player.
- Trivial PINs (1111, 1234, ...) are refused when you create or change one.

What it does **not** do:

- It does not protect against someone who copies the database file: a 4 to 8 digit PIN can be
  guessed offline. Keep `data/aiword.db` private if that matters.
- There is no email and no recovery by the player. Anyone deliberately entering wrong PINs can
  lock a profile for a few minutes.
- If a PIN is forgotten, whoever runs the game clears it with
  `python scripts/reset_pin.py Sam` (progress is kept; the next PIN typed for that name becomes
  the new one, so ask the owner to log in right away).

**Account** (sidebar) lets you change your PIN and **delete all your saved data** (both need the PIN).
Profiles created by an earlier version without a PIN keep their progress; the PIN typed at their
next login becomes their PIN.

Saved in a local SQLite file, `data/aiword.db` (standard library only, no setup); set `AIWORD_DB`
to keep it elsewhere. If the database is unavailable the game still works and says progress is
not saved.

**Updating the game:** extracting a new zip over the old folder replaces `data/puzzles.json`
(including puzzles the AI added). Before updating, copy `data/aiword.db` and `data/puzzles.json`
somewhere safe, then copy them back afterwards.

## Putting it online

**Public mode.** Set `AIWORD_PUBLIC_MODE=1` for anything on the open internet. Strangers then cannot
use up your AI key or fill the puzzle bank: custom topics, the agent, live AI puzzle generation and
auto-saving of AI puzzles are all switched off (puzzles come from the reviewed bank), and AI hints +
feedback are capped per visitor (`AIWORD_AI_SESSION_LIMIT`, default 15; the AI calls are counted when
they are *made*, even if the reply is unusable). The login page shows an honest demo notice.

**Streamlit Community Cloud (free):**

1. Push this folder to a GitHub repository (never commit `.streamlit/secrets.toml`; it is git-ignored).
2. On share.streamlit.io choose *Create app*, pick the repository and branch, and set the main file
   to `app.py`.
3. Under *Advanced settings*, choose Python 3.11 or 3.12 and paste the contents of
   `.streamlit/secrets.toml.example` into *Secrets*. To turn on AI hints and feedback, also add
   `GROQ_API_KEY = "your-key"` there. Secrets are copied into the environment automatically.
4. Deploy. Later commits to the branch update the app.

Things to know about a free hosted copy:

- **Saved profiles are not guaranteed to last.** The platform can clear local files at any time, so
  the database and any puzzles saved at runtime may vanish on a restart. Fine for a demo; for real
  persistence you need a hosted database (not built yet).
- A shared free AI key has a small daily allowance (for example 1,000 requests a day per Groq model).
  The per-visitor cap protects it, and the game falls back to built-in hints when it runs out.
- **Not yet done, worth doing before you promote it widely:** a review step so AI-written puzzles are
  approved by a person before joining the bank, and a profanity filter for the dictionary.

**Docker:** `docker build -t aiword .` then
`docker run -p 8501:8501 -e AIWORD_PUBLIC_MODE=1 -v aiword-data:/app/data aiword`
(add `-e GROQ_API_KEY=...` for AI; the key is never baked into the image). The Dockerfile has not
been built in the environment this project was developed in, so treat the first build as a test.

**Tests on every push:** `.github/workflows/tests.yml` runs the whole test suite on Python 3.11 and
3.12 with GitHub Actions. It needs no keys.

## The AI agent

With the AI connected, a sidebar switch turns on **the agent** (off by default, see the token budget below). Instead of
asking the model for a puzzle in one shot, the agent works in a loop:

1. The LLM picks a candidate word and chooses a **tool** to call.
2. Code runs the tool and feeds the real result back: `check_word`, `check_duplicate`,
   `check_leak`, `estimate_difficulty`, `check_distractors`, `lookup_definition`.
3. The LLM revises until it calls `submit_puzzle`.
4. A **validation gate** in code accepts or rejects the submission and says exactly what to fix.
5. A separate **critic** reviews the puzzle. It judges **every wrong option on its own** ("could
   this word also fit the sentence?"), because one broad question is too easily answered "none".
   The reviewer is by default a **different, larger model** (`AIWORD_CRITIC_MODEL`, default
   `openai/gpt-oss-120b`) since a model shares its own blind spots; if your key cannot use it, the
   main model reviews instead. If the critic keeps rejecting a puzzle (more than twice) the run
   **fails and the game uses another source**: a puzzle the critic rejected is never accepted.

The agent is also told what makes a fair multiple-choice puzzle: the sentence needs a clue that
only the answer satisfies (a vague "was called off due to rain" fits soccer, tennis and hockey).

Guarantees, all covered by tests: the model never validates its own work; a puzzle is accepted
only through the same strict checks as every other puzzle; bad JSON, unknown tools, weird
arguments, a dead LLM, and running out of steps or time all end safely, and the game falls back
to the one-shot generator and then the puzzle bank. Difficulty is checked with word frequency
(agrees with the hand-labelled bank 71% exactly, 95% within one level).

After each round, **"How the agent built this puzzle"** shows every step (hidden during the round
because it contains the answer). Every run is logged to `data/agent_runs.jsonl`, and the sidebar
**Agent quality** panel shows success rate, average steps, AI calls, time, and the most common
rejection reasons.

**The token budget (why the agent is opt-in).** Free plans limit tokens per minute (Groq's
`openai/gpt-oss-20b`: 8,000), and a request counts with the reply room it reserves. The agent
makes several calls per puzzle, so it is kept lean: a batch `check_words` tool checks up to 10
candidates in one call, older steps are summarised in one line (the prompt plateaus around 850
tokens instead of growing), and each step asks for a small reply. The game also tracks its own
usage and waits briefly rather than being refused (`AIWORD_API_TPM`, default 8000 on Groq, 0 = off;
`AIWORD_API_PACING_MAX_WAIT`). On a short rate-limit error it waits and retries once (up to
`AIWORD_API_RETRY_MAX_WAIT`, default 15 s); longer limits fall back to the puzzle bank. Even so, on a
free plan the agent builds roughly one puzzle a minute, so the sidebar switch is **off by default**:
use the agent to build the bank ahead of time (`python scripts/build_puzzle_bank.py --agent`, which
waits out rate limits) and live play stays instant.

Three more details: the prompt lists only the words already used **in this theme** (not all
puzzle words); the pacing budget is kept **per model**, so the reviewer's calls do not use up the
writer's minute; and every run reports the **tokens the provider actually counted**
(`try_agent.py`, the run log, and the sidebar **Agent quality** panel), so you can tune with numbers.

**If AI calls fail:** run `python scripts/check_ai.py` (add `--limits` to see every model's real
rate limits and pick the one with the highest tokens per minute). It lists the models your key can use, says
whether the configured one still exists (Groq retires models regularly), and explains any key or
address problem without ever printing the key. Pick a model with `set "AIWORD_API_MODEL=<name>"`
(Windows) or `export AIWORD_API_MODEL=<name>`. Reasoning models such as `openai/gpt-oss-20b` are
supported: they get extra token room, and their built-in tool-call format is understood.

Try it in the terminal and watch each step:

```bash
python scripts/try_agent.py "Formula 1" Easy
```

Grow the puzzle bank with the agent (slower but self-checking):

```bash
python scripts/build_puzzle_bank.py --agent --target 10
```

Not yet done: the agent has been tested with scripted LLM transcripts (including the mistakes a
small model makes), not with a large number of real model runs. Use `try_agent.py` and the quality
panel to see how your chosen model behaves and tune `AGENT_MAX_STEPS` and the prompt if needed.

## Categories and custom topics

18 built-in categories: Technology, Science, Healthcare, Finance, Environment, Space,
Engineering, Education, History, Arts & Culture, **Cricket, Entertainment, Sports,
Food & Cooking, Music, Geography, Animals, Mathematics**. Each has puzzles at all three
difficulty levels.

Choose **Custom topic...** to type anything (Formula 1, Bollywood, Chess...). The AI then
builds puzzles for that topic, checked by the same strict rules as everything else. Valid
puzzles are saved to the bank, so a topic you have used before also works without AI.

- Custom topics need the AI connected (or the topic must already exist in the bank).
  Otherwise Start is disabled and you are told why.
- If the AI fails mid-game you are told, and shown a puzzle from another category instead
  of something pretending to match your topic.
- Typed topics are sanitised (letters, digits, spaces, `& ' -` only, 40 characters max)
  because they are inserted into AI prompts.
- To add a permanent category: add it to `SECTORS` in `config.py`, then run
  `python scripts/build_puzzle_bank.py --themes "Your Category" --target 10`.

## Real words only, no duplicates

Every word comes from a strict dictionary (`data/lexicon.tsv`, built from WordNet and word
frequencies), never from what an AI merely produced:

- typos, names, brands, abbreviations and random strings are rejected
- puzzle answers must be base-form, non-obscure words (no plurals / verb forms)
- player guesses may be any real word, including plurals and verb forms
- **duplicates include inflections**: SOCKET and SOCKETS count as the same word
- hints and sentences must not leak the answer, even as a plural or inside a compound
  (rocket / rockets / rocketship)
- every wrong MCQ option must also be a real word

Rebuild the dictionary with `pip install -r requirements-build.txt` then
`python scripts/build_lexicon.py` (the built files are already included).

Difficulty adapts to how you play (3 correct in a row = harder, 2 wrong = easier).

## Where puzzles come from (hybrid)

Puzzles are tried in this order, and **whenever one source fails, the next takes over**:

1. **AI, live** - a fresh puzzle generated on demand (only if AI is connected)
2. **Puzzle bank** - `data/puzzles.json`
3. **Built-in puzzles** - last resort, so the game never breaks

Every AI puzzle goes through the same strict validation as the bank (real 6-letter word,
no answer leaked in the hint or sentence, exactly 3 distractors). Valid AI puzzles are also
**saved into the bank**, so it grows as people play. Each puzzle shows its source
(AI / bank / built-in) under the title.

To change the order, e.g. bank first: `export AIWORD_PUZZLE_ORDER=bank,ai,local`.
To stop saving AI puzzles: `export AIWORD_AUTOSAVE=0`.

## How AI is used

AI powers four features, and the game still works with none of them:

0. **Live puzzle generation** (above).

1. **Extra hint button** - asks the AI for a fresh clue that never reveals the word.
   Without AI it falls back to simple letter clues.
2. **End-of-session feedback** - a short summary of your accuracy, strongest and weakest
   themes, and a tip. Without AI you get an automatic summary.
3. **Puzzle-bank builder** - `scripts/build_puzzle_bank.py` asks the AI for new puzzles and
   keeps only ones that pass strict validation (real 6-letter word, no answer leaks in the
   hint or sentence, exactly 3 distractors).

Every AI call can fail safely: if the AI is off, rate-limited or offline, the game keeps working.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Connecting AI (free options)

Set an environment variable - never put keys in the code or commit them.

**Groq (hosted, free tier):**

```bash
export GROQ_API_KEY="your-key"        # Windows PowerShell: $env:GROQ_API_KEY="your-key"
```

The provider is detected automatically. Any OpenAI-compatible service works via
`AIWORD_API_URL`, `AIWORD_API_MODEL` and `AIWORD_API_KEY`. Model names change over time -
if you get errors, check your provider's model list and set `AIWORD_API_MODEL`.

**Local GPU model:**

```bash
pip install -r requirements-local.txt
python scripts/check_gpu.py                 # confirm CUDA works
export AIWORD_PROVIDER=local
export AIWORD_LOCAL_MODEL="Qwen/Qwen3-4B"   # optional, this is the default
```

## Growing the puzzle bank

```bash
python scripts/build_puzzle_bank.py --target 10
python scripts/build_puzzle_bank.py --themes "Space,History" --target 15
```

`--target` is the minimum puzzles per theme and difficulty; the script tops up what is missing
and saves after every theme, so an interruption never loses progress.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Structure

```
app.py                  Streamlit UI
config.py               sectors, difficulty, AI settings
ai/                     llm_client, live_generator, hints, puzzle_bank, puzzle_manager,
                        bank_builder, validator
agent/                  puzzle_agent (LLM+tools loop), tools, critic, service, log
database/               models, store (saved profiles and round history)
nlp/lexicon.py          dictionary, lemmas, duplicates, leak detection, anagram index
game/                   scoring, difficulty, feedback, anagram, star, topics, answer checking
database/models.py      Puzzle, Player, GameResult
data/puzzles.json       the puzzle bank
data/lexicon.tsv        60k-word strict dictionary (+ definitions.tsv)
scripts/                build_puzzle_bank.py, try_agent.py, check_ai.py, reset_pin.py, build_lexicon.py, check_gpu.py
tests/                  pytest suite
```
