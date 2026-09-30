"""Guards on the files that put the game online - and on not leaking secrets."""

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = [
    r"gsk_[A-Za-z0-9]{20,}",                 # Groq
    r"sk-[A-Za-z0-9_-]{20,}",                # OpenAI-style
    r"AKIA[0-9A-Z]{16}",                     # AWS access key id
    r"gh[pousr]_[A-Za-z0-9]{30,}",           # GitHub tokens
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def code_lines(text):
    """Lines that are not comments."""
    return [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]


# ---------------- no secrets, ever ----------------

def test_no_file_in_the_project_contains_anything_shaped_like_a_real_key():
    hits = []

    for path in ROOT.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
            continue

        if path.suffix in (".pyc", ".db", ".zip"):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text):
                hits.append(f"{path.relative_to(ROOT)}  matches {pattern}")

    assert hits == []


def test_gitignore_keeps_secrets_databases_and_logs_out_of_the_repository():
    ignored = read(".gitignore").splitlines()

    for needed in [".env", ".streamlit/secrets.toml", "data/aiword.db*", "data/agent_runs.jsonl", "__pycache__/"]:
        assert needed in ignored, f"{needed} is not git-ignored"


def test_docker_context_also_excludes_secrets_and_data():
    ignored = read(".dockerignore").splitlines()

    for needed in [".env", ".streamlit/secrets.toml", "data/aiword.db*", ".git"]:
        assert needed in ignored


# ---------------- the secrets template ----------------

def test_the_secrets_example_is_valid_toml_defaults_to_public_mode_and_holds_no_key():
    text = read(".streamlit/secrets.toml.example")
    settings = tomllib.loads(text)

    assert settings["AIWORD_PUBLIC_MODE"] == "1"
    assert all(isinstance(v, str) for v in settings.values())            # the bridge only copies plain text
    assert not any("GROQ_API_KEY" in line for line in code_lines(text))  # the key line stays commented out
    assert "paste-your-key-here" in text and "NEVER commit" in text


# ---------------- the container ----------------

def test_dockerfile_runs_streamlit_on_the_right_port_and_never_bakes_in_a_key():
    text = read("Dockerfile")

    assert "EXPOSE 8501" in text and "--server.address=0.0.0.0" in text and "app.py" in text
    assert "requirements.txt" in text

    for line in code_lines(text):
        if line.split()[0] in ("ENV", "ARG"):
            assert "KEY" not in line.upper() and "SECRET" not in line.upper(), line

    assert not re.search(r"(GROQ|API)_KEY\s*=\s*\S", "\n".join(code_lines(text)))


# ---------------- continuous integration ----------------

def test_the_ci_workflow_runs_the_whole_suite_on_both_supported_python_versions():
    text = read(".github/workflows/tests.yml")

    assert "pytest" in text and "requirements-dev.txt" in text
    assert '"3.11"' in text and '"3.12"' in text
    assert "on:" in text and "push" in text and "pull_request" in text
    assert "secrets." not in text                                        # CI needs no keys: tests use fakes


def test_the_dev_requirements_include_pytest_and_the_runtime_requirements():
    dev = read("requirements-dev.txt")

    assert "-r requirements.txt" in dev and "pytest" in dev


# ---------------- dependencies ----------------

def _third_party_imports():
    project = {p.name for p in ROOT.iterdir() if p.is_dir()} | {p.stem for p in ROOT.glob("*.py")}
    found = set()

    files = [ROOT / "app.py", ROOT / "config.py"]

    for folder in ("ai", "agent", "game", "database", "nlp"):
        files += list((ROOT / folder).rglob("*.py"))

    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []

            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]

            for name in names:
                if name not in sys.stdlib_module_names and name not in project:
                    found.add(name)

    return found


def test_every_library_the_app_imports_is_listed_in_a_requirements_file():
    listed = " ".join(read(n).lower().replace("_", "-") for n in
                      ("requirements.txt", "requirements-local.txt", "requirements-build.txt", "requirements-dev.txt"))

    missing = [name for name in _third_party_imports() if name.lower().replace("_", "-") not in listed]

    assert missing == []


def test_the_hosted_runtime_needs_only_streamlit_and_requests():
    runtime = read("requirements.txt").lower()

    assert "streamlit" in runtime and "requests" in runtime
    assert "english-words" not in runtime                                # no longer used: keep the install small
    assert all(heavy not in runtime for heavy in ("torch", "transformers", "nltk", "wordfreq"))


def test_the_app_never_imports_the_heavy_optional_libraries_at_startup():
    for folder in ("ai", "agent", "game", "database", "nlp"):
        for path in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))

            for node in tree.body:                                       # top level only
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name

                    assert (module or "").split(".")[0] not in ("torch", "transformers", "nltk", "wordfreq"), \
                        f"{path.name} imports {module} at startup"
