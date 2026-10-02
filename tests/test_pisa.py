# Fast unit tests — no trained models, network or API key needed.  Run: pytest -q
import copy
import json
import os
import sys

import duckdb
import httpx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_models
import warehouse_db
from drift import psi, compare


# ── Engine 3: newsvendor ─────────────────────────────────────
def test_newsvendor_orders_below_mean_when_spoilage_costs_more():
    assert ml_models.compute_optimal_order(100, 20, cu=0.25, co=1.0) < 100
    assert ml_models.compute_optimal_order(100, 20, cu=1.0, co=1.0) == 100
    assert ml_models.compute_optimal_order(1, 50) == 0  # never negative


# ── Engine 1: lag features stay within one series ────────────
def test_lag_features_use_previous_day_of_same_series():
    dates = pd.date_range("2024-01-01", periods=40)
    df = pd.DataFrame({"date": dates, "actual_demand_kg": np.arange(40.0)})
    feats = ml_models.build_lag_features(df)
    assert (feats["lag_1"] == feats["actual_demand_kg"] - 1).all()
    assert (feats["lag_7"] == feats["actual_demand_kg"] - 7).all()


def test_risk_levels():
    assert [ml_models.risk_level(s) for s in (95, 80, 79, 60, 45, 0)] == \
           ["CRITICAL", "CRITICAL", "HIGH", "HIGH", "MEDIUM", "LOW"]


# ── Drift ────────────────────────────────────────────────────
def test_psi_flags_shift_only():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.05
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.2
    rows = compare(pd.DataFrame({"x": ref}), pd.DataFrame({"x": rng.normal(1.5, 1, 500)}), ["x"])
    assert rows[0]["drifted"]


# ── Warehouse: LLM-facing connection is read-only and sandboxed ──
@pytest.fixture
def tiny_db(tmp_path):
    path = str(tmp_path / "t.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE dim_sku AS SELECT 'SKU001' AS sku_id, 'Spinach' AS sku_name")
    con.close()
    return path


def test_readonly_query_works(tiny_db):
    assert warehouse_db.query("SELECT sku_name FROM dim_sku", tiny_db).iloc[0, 0] == "Spinach"
    assert "dim_sku(sku_id VARCHAR" in warehouse_db.describe_schema(tiny_db)


@pytest.mark.parametrize("sql", [
    "DROP TABLE dim_sku",
    "INSERT INTO dim_sku VALUES ('x', 'y')",
    "SELECT * FROM read_csv_auto('config.py')",
    "COPY dim_sku TO 'leak.csv'",
])
def test_readonly_query_blocks_writes_and_file_access(tiny_db, sql):
    with pytest.raises(duckdb.Error):
        warehouse_db.query(sql, tiny_db)


# ── API ──────────────────────────────────────────────────────
def test_api_newsvendor_and_auth(monkeypatch):
    from fastapi.testclient import TestClient
    import api
    client = TestClient(api.app)

    body = {"mean_demand_kg": 50, "std_demand_kg": 10}
    r = client.post("/newsvendor", json=body)
    assert r.status_code == 200 and r.json()["critical_ratio"] == 0.2

    monkeypatch.setenv("PISA_API_KEY", "secret")
    assert client.post("/newsvendor", json=body).status_code == 401
    assert client.post("/newsvendor", json=body, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.post("/newsvendor", json={"mean_demand_kg": -1, "std_demand_kg": 1},
                       headers={"X-API-Key": "secret"}).status_code == 422


# ── Analyst bot: tool loop against a mocked Claude API ───────
def test_analyst_runs_sql_tool_then_answers(monkeypatch):
    import anthropic
    import analyst

    monkeypatch.setattr(analyst, "describe_schema", lambda db_path: "dim_vendor(vendor_id VARCHAR)")
    monkeypatch.setattr(analyst, "query", lambda sql, max_rows: pd.DataFrame({"vendor_id": ["V10"]}))
    requests_seen = []

    def handler(request):
        body = json.loads(request.content)
        requests_seen.append(body)
        usage = {"input_tokens": 10, "output_tokens": 10}
        has_result = any(isinstance(m["content"], list) and
                         any(b.get("type") == "tool_result" for b in m["content"])
                         for m in body["messages"])
        if not has_result:
            content = [{"type": "tool_use", "id": "toolu_1", "name": "run_sql",
                        "input": {"sql": "SELECT vendor_id FROM dim_vendor"}}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": "V10 (Coastal Catch) spoils most."}]
            stop = "end_turn"
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
            "content": content, "stop_reason": stop, "stop_sequence": None, "usage": usage})

    client = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    answer, sqls = analyst.ask("Worst vendor?", client=client)

    assert answer == "V10 (Coastal Catch) spoils most."
    assert sqls == ["SELECT vendor_id FROM dim_vendor"]
    assert requests_seen[0]["model"] == "claude-opus-5-5"
    assert requests_seen[0]["fallbacks"] == "default"
    tool_result = requests_seen[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and "V10" in str(tool_result["content"])


# ── Analyst bot: Groq path + provider selection (no network) ──
class _FakeResp:
    def __init__(self, body, status=200):
        self._body, self.status_code, self.ok = body, status, status < 400
        self.text = json.dumps(body)

    def json(self):
        return self._body


def _groq_fake(monkeypatch, replies):
    """Patch requests.post in analyst to return `replies` in order; returns the list of sent payloads."""
    import analyst
    sent, queue = [], list(replies)
    monkeypatch.setattr(analyst, "describe_schema", lambda db_path: "dim_vendor(vendor_id VARCHAR)")
    monkeypatch.setattr(analyst, "query", lambda sql, max_rows: pd.DataFrame({"vendor_id": ["V10"]}))
    monkeypatch.setattr(analyst.requests, "post",
                        lambda url, headers, json, timeout: (sent.append({"headers": headers, "json": copy.deepcopy(json)}),
                                                             queue.pop(0) if len(queue) > 1 else queue[0])[1])
    return sent


def _tool_call_reply(call_id="c1", sql="SELECT vendor_id FROM dim_vendor"):
    return _FakeResp({"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "reasoning": "internal",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "run_sql", "arguments": json.dumps({"sql": sql})}}]}}]})


def _answer_reply(content):
    return _FakeResp({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]})


@pytest.fixture(autouse=True)
def _no_llm_keys(monkeypatch):
    # Tests must not depend on keys set on the dev machine or in CI
    for k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_groq_runs_tool_then_answers(monkeypatch):
    import analyst
    sent = _groq_fake(monkeypatch, [_tool_call_reply(), _answer_reply("V10 spoils most.")])

    answer, sqls = analyst.ask("Worst vendor?", api_key="gsk_test")

    assert answer == "V10 spoils most."
    assert sqls == ["SELECT vendor_id FROM dim_vendor"]
    assert sent[0]["headers"]["Authorization"] == "Bearer gsk_test"
    second = sent[1]["json"]["messages"]
    assert "reasoning" not in second[-2]                     # only standard fields sent back
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1" and "V10" in second[-1]["content"]


def test_groq_null_content_and_round_limit(monkeypatch):
    import analyst
    _groq_fake(monkeypatch, [_answer_reply(None)])
    assert analyst.ask("q", api_key="gsk_test")[0] == "I couldn't find an answer in the data."

    sent = _groq_fake(monkeypatch, [_tool_call_reply()])     # model never stops calling tools
    answer, sqls = analyst.ask("q", api_key="gsk_test")
    assert len(sent) == analyst.MAX_TOOL_ROUNDS
    assert "couldn't finish" in answer and "V10" not in answer   # never the raw CSV


def test_groq_rejected_key_is_an_auth_error(monkeypatch):
    import analyst
    _groq_fake(monkeypatch, [_FakeResp({"error": {"message": "Invalid API Key"}}, status=401)])
    with pytest.raises(analyst.LLMError) as e:
        analyst.ask("q", api_key="gsk_bad")
    assert e.value.auth


@pytest.mark.parametrize("key, engine", [("gsk_x", "groq"), ("sk-ant-x", "claude")])
def test_ask_routes_by_key_prefix(monkeypatch, key, engine):
    import analyst
    monkeypatch.setattr(analyst, "_ask_groq", lambda *a, **kw: ("groq", []))
    monkeypatch.setattr(analyst, "_ask_claude", lambda *a, **kw: ("claude", []))
    assert analyst.ask("q", api_key=key)[0] == engine


def test_ask_rejects_missing_or_unknown_key(monkeypatch):
    import analyst
    monkeypatch.setattr(analyst, "resolve_key", lambda api_key=None: api_key)
    for key in (None, "not-a-key"):
        with pytest.raises(analyst.LLMError):
            analyst.ask("q", api_key=key)


# ── Security controls ────────────────────────────────────────
def test_readonly_sql_cannot_unlock_its_own_limits(tiny_db):
    for sql in ("SET memory_limit='8GB'", "SET enable_external_access=true", "ATTACH 'x.db'", "INSTALL httpfs"):
        with pytest.raises(duckdb.Error):
            warehouse_db.query(sql, tiny_db)


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    import api
    api._hits.clear()
    return TestClient(api.app), api


BODY = {"mean_demand_kg": 50, "std_demand_kg": 10}


def test_api_fails_closed_in_production_without_key(monkeypatch, api_client):
    client, _ = api_client
    monkeypatch.delenv("PISA_API_KEY", raising=False)
    monkeypatch.setenv("PISA_ENV", "production")
    assert client.post("/newsvendor", json=BODY).status_code == 503


def test_api_rejects_unknown_fields_and_sets_security_headers(monkeypatch, api_client):
    client, _ = api_client
    monkeypatch.delenv("PISA_ENV", raising=False)
    r = client.post("/newsvendor", json={**BODY, "is_admin": True})
    assert r.status_code == 422
    r = client.post("/newsvendor", json=BODY)
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"


def test_api_rate_limits_per_client(monkeypatch, api_client):
    client, api = api_client
    monkeypatch.delenv("PISA_ENV", raising=False)
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MIN", 3)
    codes = [client.post("/newsvendor", json=BODY).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
