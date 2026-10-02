# =============================================================
# analyst.py — "Ask PISA" LLM analyst (Groq + Claude API + DuckDB)
#
# Answers plain-English questions by writing DuckDB SQL against the
# warehouse star schema through a run_sql tool. Queries run on a
# read-only connection with no file/network access (warehouse_db).
#
# Supports:
#   - Groq API (gsk_...) with ultra-fast inference (default)
#   - Anthropic Claude (sk-ant-...)
#
# CLI: python analyst.py "Which vendor has the worst spoilage rate?"
# =============================================================

import json
import os
import sys
import requests

from config import CLAUDE_MODEL, DB_PATH
from warehouse_db import describe_schema, query

MAX_RESULT_ROWS = 50

SYSTEM_PROMPT = """You are PISA's supply-chain analyst for Zomato Hyperpure's B2B perishables warehouses.
Answer questions about demand, spoilage/wastage, cold-chain sensors, vendors and live inventory risk
by querying the DuckDB warehouse with the run_sql tool. Use DuckDB SQL dialect.

Prefer the mart_* views when they fit; join facts to dims for names. Money is in INR — format
large values in lakhs (₹1L = ₹100,000). fact_active_lots holds today's stock with model risk
scores (risk_score 0-100, risk_level, spoil_prob_48h = % chance of spoiling in the next 48h).

Answer in a few sentences or a small markdown table, cite the numbers you found, and say plainly
if the data cannot answer the question. Do not show the SQL unless asked.

Schema:
{schema}"""


from anthropic import beta_tool

@beta_tool
def run_sql(sql: str) -> str:
    """Run a read-only DuckDB SQL query against the PISA warehouse and return up to 50 rows as CSV.

    Args:
        sql: A single SELECT (or WITH ... SELECT) statement in DuckDB dialect.
    """
    try:
        df = query(sql, max_rows=MAX_RESULT_ROWS + 1)
    except Exception as e:
        return f"ERROR: {e}"
    note = f"\n(truncated to {MAX_RESULT_ROWS} rows)" if len(df) > MAX_RESULT_ROWS else ""
    return df.head(MAX_RESULT_ROWS).to_csv(index=False) + note


def _ask_groq(question, history=None, db_path=DB_PATH, api_key=None):
    """Answer using Groq API (OpenAI-compatible) with tool use."""
    tools = [{
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run a read-only DuckDB SQL query against the PISA warehouse and return up to 50 rows as CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SELECT (or WITH ... SELECT) statement in DuckDB SQL dialect."
                    }
                },
                "required": ["sql"]
            }
        }
    }]
    system_msg = {"role": "system", "content": SYSTEM_PROMPT.format(schema=describe_schema(db_path))}
    messages = [system_msg]
    if history:
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    sqls_run = []
    for _ in range(5):
        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": messages,
            "tools": tools,
            "temperature": 0.1,
            "max_tokens": 1500,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if choice.get("finish_reason") == "tool_calls" or tool_calls:
            for tc in (tool_calls or []):
                fn = tc.get("function", {})
                if fn.get("name") == "run_sql":
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {"sql": args}
                    query_str = args.get("sql", "")
                    sqls_run.append(query_str)
                    result_csv = run_sql(query_str)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "name": "run_sql",
                        "content": result_csv
                    })
        else:
            answer = msg.get("content", "").strip()
            return answer or "I couldn't find an answer in the data.", sqls_run

    return messages[-1].get("content", "Completed query execution."), sqls_run


def _ask_claude(question, history=None, db_path=DB_PATH, client=None):
    """Answer using Anthropic Claude tool runner."""
    client = client or anthropic.Anthropic()
    messages = list(history or []) + [{"role": "user", "content": question}]

    runner = client.beta.messages.tool_runner(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT.format(schema=describe_schema(db_path)),
        tools=[run_sql],
        messages=messages,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    sql_run, final = [], None
    for message in runner:
        final = message
        sql_run += [b.input["sql"] for b in message.content
                    if b.type == "tool_use" and b.name == "run_sql"]

    if final is None or final.stop_reason == "refusal":
        return "Sorry, I can't help with that question.", sql_run
    answer = "".join(b.text for b in final.content if b.type == "text").strip()
    return answer or "I couldn't find an answer in the data.", sql_run


def ask(question, history=None, db_path=DB_PATH, api_key=None, client=None):
    """Answers one question. Supports Groq or Claude depending on key provided."""
    key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    if not key and not client:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass

    if key and key.startswith("gsk_"):
        return _ask_groq(question, history=history, db_path=db_path, api_key=key)

    return _ask_claude(question, history=history, db_path=db_path, client=client)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    q = " ".join(sys.argv[1:]) or "Which vendor has the highest spoilage rate, and what did it cost us?"
    answer, sqls = ask(q)
    for s in sqls:
        print(f"-- SQL:\n{s}\n")
    print(answer)
