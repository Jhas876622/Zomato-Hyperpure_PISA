# =============================================================
# analyst.py — "Ask PISA" LLM analyst (Groq + Claude API + DuckDB)
#
# Answers plain-English questions by writing DuckDB SQL against the
# warehouse star schema through a run_sql tool. Queries run on a
# read-only connection with no file/network access (warehouse_db).
#
# Engine is picked from the API key:
#   - gsk_...    → Groq (config.GROQ_MODEL), the default
#   - sk-ant-... → Anthropic Claude (config.CLAUDE_MODEL)
# Key lookup order: api_key argument → GROQ_API_KEY → ANTHROPIC_API_KEY
# (env, then Streamlit secrets). Passing client= always uses Claude.
#
# CLI: python analyst.py "Which vendor has the worst spoilage rate?"
# =============================================================

import json
import os
import sys

import anthropic
import requests
from anthropic import beta_tool

from config import CLAUDE_MODEL, DB_PATH, GROQ_MODEL
from warehouse_db import describe_schema, query

MAX_RESULT_ROWS = 50
MAX_TOOL_ROUNDS = 5
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are PISA's AI Supply-Chain Analyst & Copilot for Zomato Hyperpure's B2B perishable operations.
Answer questions about demand forecasts, spoilage/wastage risk, cold-storage IoT sensors, vendor quality, and live inventory.
Write DuckDB SQL queries with the run_sql tool to find exact numbers from the warehouse database.

Language & Tone:
- You fluently understand both English and Hinglish / Hindi (e.g. "sabse bekar vendor kaunsa hai?", "kitna maal kharab ho raha hai?").
- If the user asks in Hinglish or Hindi, reply in professional, easy-to-read Hinglish. If in English, reply in English.
- If asked to draft a message (e.g. "Draft WhatsApp alert for warehouse manager" or "Draft email"), format it cleanly with emojis and key action items ready to copy and send.

Database Guidelines:
- Useful tables and views:
  * `mart_vendor_scorecard`: vendor_id, vendor_name, spoil_rate_pct, spoiled_value_inr, quality_reject_rate, on_time_rate.
  * `fact_active_lots`: today's stock, scored by the nightly pipeline: lot_id, sku_name, warehouse_name, quantity_kg,
    lot_value_inr, risk_score (0-100), risk_level ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW'),
    spoil_prob_48h (% chance of spoiling within 48 hours, 0-100), recommended_action.
  * `mart_daily_demand`: date, city, category, demand_kg, revenue_inr.
  * `mart_wastage_by_category_month`: monthly category wastage %, procured_value_inr, spoiled_value_inr.
  * `mart_cold_chain_excursions`: per warehouse, zone and day: max_temp_c, avg_temp_c,
    hours_above_setpoint_3c (hours more than 3°C above the zone setpoint).
- If a column is missing, check the schema below instead of guessing.
- Money is in INR (₹) — format amounts in lakhs (e.g. ₹1.5L = ₹150,000) or thousands.
- Cite numbers and percentages directly. Present comparative results in clear markdown tables.
- Do not show raw SQL in the answer unless specifically asked.

Schema:
{schema}"""


class LLMError(Exception):
    """A problem the user can act on (bad key, rate limit, provider outage). Message is UI-ready."""
    def __init__(self, message, auth=False):
        super().__init__(message)
        self.auth = auth  # True → the key itself was rejected; the UI should ask for a new one


@beta_tool
def run_sql(sql: str) -> str:
    """Run a read-only DuckDB SQL query against the PISA warehouse and return up to 50 rows as CSV.

    Args:
        sql: A single SELECT (or WITH ... SELECT) statement in DuckDB dialect.
    """
    try:
        df = query(sql, max_rows=MAX_RESULT_ROWS + 1)
    except Exception as e:  # SQL errors go back to the model so it can fix the query
        return f"ERROR: {e}"
    note = f"\n(truncated to {MAX_RESULT_ROWS} rows)" if len(df) > MAX_RESULT_ROWS else ""
    return df.head(MAX_RESULT_ROWS).to_csv(index=False) + note


GROQ_TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "Run a read-only DuckDB SQL query against the PISA warehouse and return up to 50 rows as CSV.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string",
                                   "description": "A single SELECT (or WITH ... SELECT) statement in DuckDB SQL dialect."}},
            "required": ["sql"],
        },
    },
}]


def _groq_post(payload, api_key):
    try:
        resp = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30)
    except requests.Timeout:
        raise LLMError("Groq took longer than 30 seconds to answer. Please try again.")
    except requests.RequestException as e:
        raise LLMError(f"Could not reach Groq: {e}")
    if resp.ok:
        return resp.json()
    try:
        detail = resp.json().get("error", {}).get("message") or resp.text[:300]
    except ValueError:
        detail = resp.text[:300]
    if resp.status_code in (401, 403):
        raise LLMError("Groq rejected this API key. Please enter a valid key.", auth=True)
    if resp.status_code == 429:
        raise LLMError("Groq rate limit reached. Wait a minute and try again.")
    raise LLMError(f"Groq API error {resp.status_code}: {detail}")


def _run_tool_call(tc, sqls_run):
    """Executes one tool call and returns the tool message the API requires for its id."""
    fn = tc.get("function", {})
    if fn.get("name") != "run_sql":
        content = f"ERROR: unknown tool {fn.get('name')!r}; only run_sql is available."
    else:
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict) or not args.get("sql"):
            content = 'ERROR: arguments must be JSON like {"sql": "SELECT ..."}.'
        else:
            sqls_run.append(args["sql"])
            content = run_sql(args["sql"])
    return {"role": "tool", "tool_call_id": tc.get("id"), "content": content}


def _ask_groq(question, history=None, db_path=DB_PATH, api_key=None):
    """Answer using Groq's OpenAI-compatible API with a run_sql tool loop."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(schema=describe_schema(db_path))}]
    messages += [{"role": t["role"], "content": t["content"]} for t in (history or [])]
    messages.append({"role": "user", "content": question})

    sqls_run = []
    for _ in range(MAX_TOOL_ROUNDS):
        data = _groq_post({"model": GROQ_MODEL, "messages": messages, "tools": GROQ_TOOLS,
                           "temperature": 0.1, "max_tokens": 1500}, api_key)
        choice = data["choices"][0]
        msg = choice["message"]
        tool_calls = msg.get("tool_calls") or []

        # Send back only standard fields; extras like "reasoning" can be rejected on the next call
        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)

        if not tool_calls:
            answer = (msg.get("content") or "").strip()
            if choice.get("finish_reason") == "length":
                answer += "\n\n_(Answer was cut off at the length limit; ask a narrower question for the rest.)_"
            return answer or "I couldn't find an answer in the data.", sqls_run

        messages += [_run_tool_call(tc, sqls_run) for tc in tool_calls]

    return (f"I couldn't finish this within {MAX_TOOL_ROUNDS} query steps. "
            "Try a narrower question, e.g. one hub or one category."), sqls_run


def _ask_claude(question, history=None, db_path=DB_PATH, client=None, api_key=None):
    """Answer using the Anthropic tool runner."""
    client = client or anthropic.Anthropic(api_key=api_key)
    messages = list(history or []) + [{"role": "user", "content": question}]

    runner = client.beta.messages.tool_runner(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT.format(schema=describe_schema(db_path)),
        tools=[run_sql],
        messages=messages,
        output_config={"effort": "medium"},
        # On a safety decline, the API re-runs the request on a fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    sql_run, final = [], None
    try:
        for message in runner:
            final = message
            sql_run += [b.input["sql"] for b in message.content
                        if b.type == "tool_use" and b.name == "run_sql"]
    except anthropic.AuthenticationError:
        raise LLMError("Anthropic rejected this API key. Please enter a valid key.", auth=True)
    except anthropic.RateLimitError:
        raise LLMError("Anthropic rate limit reached. Wait a minute and try again.")
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Could not reach Anthropic: {e}")
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}")

    if final is None or final.stop_reason == "refusal":
        return "Sorry, I can't help with that question.", sql_run
    answer = "".join(b.text for b in final.content if b.type == "text").strip()
    return answer or "I couldn't find an answer in the data.", sql_run


def resolve_key(api_key=None):
    """api_key argument → GROQ/ANTHROPIC env vars → Streamlit secrets. Returns None if nothing is set."""
    key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
        except FileNotFoundError:  # no secrets.toml — fine outside Streamlit Cloud
            pass
    return key.strip() if key else None


def ask(question, history=None, db_path=DB_PATH, api_key=None, client=None):
    """Answers one question. Returns (answer_text, list_of_sql_run). Raises LLMError on key/API problems."""
    if client is not None:
        return _ask_claude(question, history=history, db_path=db_path, client=client)

    key = resolve_key(api_key)
    if not key:
        raise LLMError("No LLM key configured. Set GROQ_API_KEY (gsk_...) or ANTHROPIC_API_KEY (sk-ant-...).", auth=True)
    if key.startswith("gsk_"):
        return _ask_groq(question, history=history, db_path=db_path, api_key=key)
    if key.startswith("sk-ant-"):
        return _ask_claude(question, history=history, db_path=db_path, api_key=key)
    raise LLMError("Unrecognised key. Expected a Groq key (gsk_...) or an Anthropic key (sk-ant-...).", auth=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    q = " ".join(sys.argv[1:]) or "Which vendor has the highest spoilage rate, and what did it cost us?"
    try:
        answer, sqls = ask(q)
    except LLMError as e:
        sys.exit(f"Error: {e}")
    for s in sqls:
        print(f"-- SQL:\n{s}\n")
    print(answer)
