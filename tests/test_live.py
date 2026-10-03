# Live integration tests: real network calls that spend real API quota.
# Skipped unless RUN_LIVE_TESTS=1. Key comes from GROQ_API_KEY or .streamlit/secrets.toml.
#   RUN_LIVE_TESTS=1 pytest -q tests/test_live.py
import os
import sys
import tomllib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to run")


def _groq_key():
    if os.getenv("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    path = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return tomllib.load(f).get("GROQ_API_KEY")
    return None


def test_groq_answers_from_the_warehouse():
    import analyst
    from config import DB_PATH

    key = _groq_key()
    if not key:
        pytest.skip("no Groq key available")
    if not os.path.exists(os.path.join(ROOT, DB_PATH)):
        pytest.skip("warehouse not built — run python pipeline.py")

    os.chdir(ROOT)  # DB_PATH is relative to the project root
    answer, sqls = analyst.ask("How many vendors are in dim_vendor? Reply with just the number.", api_key=key)

    assert sqls, "the model should have queried the warehouse"
    assert "10" in answer
