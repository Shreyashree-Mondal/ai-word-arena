import os

# ---------------------------------------
# AI Word Arena Configuration
# ---------------------------------------

# Available sectors (themes)
SECTORS = [
    "Technology",
    "Science",
    "Healthcare",
    "Finance",
    "Environment",
    "Space",
    "Engineering",
    "Education",
    "History",
    "Arts & Culture",
    "Cricket",
    "Entertainment",
    "Sports",
    "Food & Cooking",
    "Music",
    "Geography",
    "Animals",
    "Mathematics",
]

# Wordle
WORD_LENGTH = 6
MAX_WORDLE_ATTEMPTS = 6

# Difficulty levels (index 0 = level 1)
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

# ---------------------------------------
# AI configuration
# ---------------------------------------
# The core game runs entirely from data/puzzles.json and never needs AI.
# AI is used for three optional extras:
#   1. the "Need another hint?" button
#   2. end-of-session feedback
#   3. building/extending the puzzle bank offline (scripts/build_puzzle_bank.py)
#
# Providers:
#   "api"   - any OpenAI-compatible chat API (Groq by default; free tier)
#   "local" - a Hugging Face model on your own GPU (needs requirements-local.txt)
#   "none"  - AI switched off; the game uses built-in fallbacks
#
# Set via environment variables so no secret is ever stored in the code:
#   export GROQ_API_KEY="..."          -> provider auto-selects "api"
#   export AIWORD_PROVIDER=local       -> use the local GPU model instead

_api_key_present = bool(os.getenv("AIWORD_API_KEY") or os.getenv("GROQ_API_KEY"))

AI_PROVIDER = (
    os.getenv("AIWORD_PROVIDER") or ("api" if _api_key_present else "none")
).lower()

# "api" provider settings (defaults point at Groq's OpenAI-compatible endpoint)
API_URL = os.getenv(
    "AIWORD_API_URL", "https://api.groq.com/openai/v1/chat/completions"
)
# Groq retires models regularly. If AI calls fail with 404, run
# `python scripts/check_ai.py` to see which models your key can use.
API_MODEL = os.getenv("AIWORD_API_MODEL", "openai/gpt-oss-20b")

# Reasoning models (gpt-oss, qwen3, ...) "think" before answering and that hidden
# thinking is counted against max_tokens, so they need extra room.
#   AIWORD_API_EXTRA_TOKENS: "auto" (1000 for reasoning models, else 0) or a number
#   AIWORD_API_REASONING:    "auto" ("low" for gpt-oss, otherwise not sent),
#                            "off", or a value your model accepts (low/medium/high)
API_EXTRA_TOKENS = os.getenv("AIWORD_API_EXTRA_TOKENS", "auto")
API_REASONING_EFFORT = os.getenv("AIWORD_API_REASONING", "auto")

# "local" provider settings
LOCAL_MODEL_NAME = os.getenv("AIWORD_LOCAL_MODEL", "Qwen/Qwen3-4B")

AI_TIMEOUT_SECONDS = 30

# On a rate-limit error (HTTP 429) wait and try once more, but only if the service says
# the wait is short. Longer waits (e.g. a daily limit) are reported instead of blocking.
API_RETRY_MAX_WAIT = float(os.getenv("AIWORD_API_RETRY_MAX_WAIT", "15"))

# Free plans cap TOKENS PER MINUTE (Groq's gpt-oss-20b: 8,000), and a request is counted
# with the reply room it reserves. The game therefore tracks its own use and waits a little
# rather than being refused. 0 turns the pacing off. Default: 8000 on Groq, off elsewhere.
API_TPM_LIMIT = int(os.getenv("AIWORD_API_TPM", "8000" if "groq.com" in API_URL else "0"))
API_PACING_MAX_WAIT = float(os.getenv("AIWORD_API_PACING_MAX_WAIT", "12"))   # scripts use longer

# Hints per puzzle
MAX_HINTS = 3


# ---------------------------------------
# Where puzzles come from
# ---------------------------------------
# Tried in this order; whenever one fails, the next one takes over.
#   "ai"    - generated live by the AI (only if AI is connected)
#   "bank"  - data/puzzles.json
#   "local" - small built-in set (last resort)
# Example: AIWORD_PUZZLE_ORDER=bank,ai,local  (bank first)
PUZZLE_ORDER = [
    name.strip()
    for name in os.getenv("AIWORD_PUZZLE_ORDER", "ai,bank,local").split(",")
    if name.strip()
]

# Save every valid AI-generated puzzle into the bank so it keeps growing
AUTO_SAVE_AI_PUZZLES = os.getenv("AIWORD_AUTOSAVE", "1") != "0"

# Live generation limits (kept short so the game never feels stuck)
LIVE_AI_TIMEOUT = 12
LIVE_MAX_ATTEMPTS = 2
LIVE_BATCH_SIZE = 3

# ---------------------------------------
# Anagram mode
# ---------------------------------------
MAX_ANAGRAM_ATTEMPTS = 3

# ---------------------------------------
# AI agent (LLM + tools loop)
# ---------------------------------------
AGENT_MAX_STEPS = 8
AGENT_MAX_TOKENS = 300            # room for one JSON action (kept small: it counts against the rate limit)
AGENT_TIME_BUDGET = 45            # seconds for one puzzle
AGENT_USE_CRITIC = os.getenv("AIWORD_AGENT_CRITIC", "1") != "0"
AGENT_MAX_CRITIC_REJECTIONS = 2   # how many times the critic may send the agent back to fix a puzzle
# The reviewer should be a different, ideally larger, model than the writer: a model
# tends to share its own blind spots. Empty = review with the same model. If this model
# is unavailable for your key, the game quietly falls back to the main model.
CRITIC_MODEL = os.getenv("AIWORD_CRITIC_MODEL", "openai/gpt-oss-120b")
AGENT_LOG_ENABLED = True
AGENT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "agent_runs.jsonl"
)

# ---------------------------------------
# Saved profiles (SQLite)
# ---------------------------------------
DB_PATH = os.getenv("AIWORD_DB") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "aiword.db"
)

# ---------------------------------------
# PIN protection for saved profiles
# ---------------------------------------
PIN_MIN_LEN = 4
PIN_MAX_LEN = 8
PIN_HASH_ITERATIONS = 600_000     # PBKDF2-HMAC-SHA256; tests lower this for speed
PIN_MAX_ATTEMPTS = 5              # wrong PINs before a profile is locked
PIN_LOCK_SECONDS = 300            # how long the lock lasts


# ---------------------------------------
# Public deployment
# ---------------------------------------
# Set AIWORD_PUBLIC_MODE=1 when the game is on the open internet. Strangers then cannot:
#   - use up your AI key's daily allowance (hints and feedback are capped per visit)
#   - fill the shared puzzle bank with their own topics or unreviewed AI puzzles
# Custom topics, the AI agent and live AI puzzle generation are switched off, and AI puzzles
# are not saved. Puzzles come from the reviewed bank; AI only adds hints and feedback.
PUBLIC_MODE = os.getenv("AIWORD_PUBLIC_MODE", "0") == "1"

# AI-written hints + feedback allowed per visitor session in public mode (0 = no limit)
AI_SESSION_LIMIT = int(os.getenv("AIWORD_AI_SESSION_LIMIT", "15"))

if PUBLIC_MODE:
    AUTO_SAVE_AI_PUZZLES = False
    PUZZLE_ORDER = [name for name in PUZZLE_ORDER if name != "ai"] or ["bank", "local"]
