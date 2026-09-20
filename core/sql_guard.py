"""Safety boundary around executing LLM-generated SQL against DuckDB.

A naive "reject anything that isn't SELECT" check is not sufficient:
`SELECT * FROM read_csv_auto('/etc/passwd')` or `read_parquet('s3://...')`
are legal, read-only SELECT statements that still escape the sandbox.

Verified empirically against sqlglot 25.x / duckdb 1.5.x (see git history /
the plan doc for the probe scripts): table-valued functions used as a scan
source show up as an `exp.Table` node whose `.this` is NOT a plain
`exp.Identifier` (it's `Anonymous` for read_csv_auto/sqlite_scan, or even a
dedicated node class like `ReadParquet` for read_parquet — dialect-specific,
so a function-name denylist is fragile). The robust check is structural:
every table reference anywhere in the query must be a bare identifier that
matches a name in the session's allow-list. Nothing else is inspected or
blocked, so legitimate aggregate/window functions in SELECT/WHERE/GROUP BY
(SUM, date_trunc, regr_slope, ...) are completely unaffected.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

DEFAULT_MEMORY_LIMIT = "512MB"
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_ROW_CAP = 500

_READ_ONLY_ROOTS = (exp.Select, exp.Union, exp.Except, exp.Intersect)


@dataclass
class ValidationResult:
    ok: bool
    error: str | None = None


@dataclass
class ExecResult:
    df: pd.DataFrame | None
    truncated: bool
    error: str | None
    row_count: int = 0


def create_sandboxed_connection() -> duckdb.DuckDBPyConnection:
    """A fresh in-memory DuckDB connection with autoload/autoinstall disabled
    and a memory cap — this is the only place a session's connection is created.
    """
    con = duckdb.connect(database=":memory:", config={"memory_limit": DEFAULT_MEMORY_LIMIT})
    con.execute("SET autoinstall_known_extensions=false")
    con.execute("SET autoload_known_extensions=false")
    return con


def validate_sql(sql: str, allowed_tables: set[str]) -> ValidationResult:
    sql = sql.strip()
    if not sql:
        return ValidationResult(False, "Empty query.")

    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except Exception as e:  # sqlglot raises various ParseError subtypes
        return ValidationResult(False, f"Could not parse SQL: {e}")

    if len(statements) != 1:
        return ValidationResult(False, "Only a single SQL statement is allowed.")

    stmt = statements[0]
    if not isinstance(stmt, _READ_ONLY_ROOTS):
        return ValidationResult(False, f"Only read-only SELECT statements are allowed (got {type(stmt).__name__}).")

    cte_names = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables} | cte_names

    for table in stmt.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier) or not table.name:
            return ValidationResult(False, f"Query references a disallowed data source: {table.sql(dialect='duckdb')}")
        if table.name.lower() not in allowed:
            return ValidationResult(False, f'Query references an unknown table: "{table.name}"')

    return ValidationResult(True)


def execute_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    row_cap: int = DEFAULT_ROW_CAP,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecResult:
    """Run pre-validated SQL with a hard timeout. Caller must call validate_sql first."""
    watchdog = threading.Timer(timeout_s, con.interrupt)
    watchdog.start()
    try:
        df = con.execute(sql).fetch_df()
    except Exception as e:
        return ExecResult(df=None, truncated=False, error=str(e))
    finally:
        watchdog.cancel()

    row_count = len(df)
    truncated = row_count > row_cap
    if truncated:
        df = df.head(row_cap)
    return ExecResult(df=df, truncated=truncated, error=None, row_count=row_count)
