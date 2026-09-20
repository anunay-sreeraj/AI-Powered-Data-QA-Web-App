"""The core "delta": translate a question to SQL, validate it, execute it,
and self-repair on error - the LLM never computes an answer itself, it only
ever produces a query that gets actually run against the real data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb
import pandas as pd

from core import llm_client, sql_guard

MAX_RETRIES = 2

SYSTEM_PROMPT = """You are a data assistant working with DuckDB tables. Translate the user's question into exactly one of two response shapes - decide which one fits based on what the question is actually asking, not on keywords.

1. If the question asks you to compute, filter, aggregate, or compare values (totals, averages, counts, trends, comparisons, etc.), output ONLY a single DuckDB SQL SELECT (or WITH ... SELECT) statement. No prose, no explanation, no markdown code fences.
2. If the question instead asks what the data/files/tables/sheets represent or contain - a descriptive question, not asking for a computed result - respond with a single line starting with "DESCRIBE:" followed by a plain-English description of the relevant table(s), based only on their names, columns, and the sample rows given below. Do not state any number or computed statistic (totals, averages, counts, etc.) in this response shape - you have not run a query, so you have no basis for any figure.

Rules for SQL (response shape 1):
- Only reference the tables and columns listed in the schema below. Never invent a table or column name.
- Always double-quote every table and column identifier, e.g. SELECT "amount" FROM "orders".
- Use DuckDB functions for statistics/trends where useful: date_trunc, regr_slope, corr, percentile_cont, stddev, window functions.
- If the question needs data from more than one table, write an explicit JOIN using columns that plausibly correspond across tables.
- Never use DDL/DML (no INSERT/UPDATE/DELETE/DROP/CREATE/ATTACH/COPY/PRAGMA) and never call file- or network-reading functions (read_csv, read_parquet, etc.) - only query the given tables directly.
"""

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)
_DESCRIBE_PREFIX = "DESCRIBE:"


def build_messages(question: str, schema_context: str, history: list[dict]) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema_context}"},
        {
            "role": "assistant",
            "content": "Understood. I will only reference these tables/columns and output raw SQL only.",
        },
    ]
    for turn in history:
        if turn.get("sql"):
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["sql"]})
    messages.append({"role": "user", "content": question})
    return messages


def extract_sql(raw: str) -> str:
    text = raw.strip()
    text = _FENCE_RE.sub("", text).strip()
    return text.rstrip(";").strip()


@dataclass
class SqlGenResult:
    sql: str | None
    df: pd.DataFrame | None
    truncated: bool
    error: str | None
    attempts: int
    row_count: int = 0
    description: str | None = None


def generate_and_execute(
    question: str,
    schema_context: str,
    history: list[dict],
    con: duckdb.DuckDBPyConnection,
    allowed_tables: set[str],
    max_retries: int = MAX_RETRIES,
) -> SqlGenResult:
    """Runs the model's response through one of two paths, chosen by the model
    itself (not by any pre-classification of the question): a DESCRIBE:-prefixed
    response is returned directly as a description, anything else is treated as
    SQL and goes through the existing validate/execute/self-repair flow.
    """
    messages = build_messages(question, schema_context, history)
    last_sql: str | None = None
    last_error: str | None = None

    for attempt in range(1, max_retries + 2):  # initial attempt + retries
        raw = llm_client.chat(messages)

        stripped = raw.strip()
        if stripped.upper().startswith(_DESCRIBE_PREFIX):
            description = stripped[len(_DESCRIBE_PREFIX):].strip()
            return SqlGenResult(sql=None, df=None, truncated=False, error=None, attempts=attempt, description=description)

        sql = extract_sql(raw)
        last_sql = sql

        validation = sql_guard.validate_sql(sql, allowed_tables)
        if not validation.ok:
            last_error = validation.error
            messages.append({"role": "assistant", "content": sql})
            messages.append(
                {"role": "user", "content": f"That query is invalid: {validation.error}\nFix it and output only the corrected SQL."}
            )
            continue

        result = sql_guard.execute_sql(con, sql)
        if result.error:
            last_error = result.error
            messages.append({"role": "assistant", "content": sql})
            messages.append(
                {"role": "user", "content": f"That query failed with error: {result.error}\nFix it and output only the corrected SQL."}
            )
            continue

        return SqlGenResult(
            sql=sql, df=result.df, truncated=result.truncated, error=None, attempts=attempt, row_count=result.row_count
        )

    return SqlGenResult(sql=last_sql, df=None, truncated=False, error=last_error, attempts=max_retries + 1)


def phrase_answer(question: str, df: pd.DataFrame | None) -> str:
    """Restate the already-computed result in one sentence. Never asked to recompute anything."""
    if df is None or df.empty:
        return "That query didn't return any rows."

    if df.shape == (1, 1):
        col = df.columns[0]
        value = df.iloc[0, 0]
        return f"**{col}**: {value}"

    messages = [
        {
            "role": "system",
            "content": (
                "You restate an already-computed data result as one or two short, plain-English sentences. "
                "You are given the exact result values - never invent, recompute, or adjust any number, "
                "just describe what the table shows."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nResult (already computed, do not recompute):\n{df.head(20).to_dict(orient='records')}",
        },
    ]
    try:
        return llm_client.chat(messages, max_tokens=200)
    except Exception:
        return f"Here are the results ({len(df)} row{'s' if len(df) != 1 else ''})."
