"""Thin wrapper around st.session_state: one sandboxed DuckDB connection,
schema-profile cache, and short Q&A history per browser session.

Streamlit reruns the whole script on every interaction, so anything that
must survive a rerun (the DB connection, ingested tables, chat history)
lives in st.session_state rather than as a plain module/local variable.
"""
from __future__ import annotations

import duckdb
import streamlit as st

from core import sql_guard

MAX_HISTORY_TURNS = 3


def init() -> None:
    if "db_con" not in st.session_state:
        st.session_state.db_con = sql_guard.create_sandboxed_connection()
    if "table_profiles" not in st.session_state:
        st.session_state.table_profiles = {}
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = set()
    if "history" not in st.session_state:
        st.session_state.history = []


def get_connection() -> duckdb.DuckDBPyConnection:
    init()
    return st.session_state.db_con


def get_profiles() -> dict[str, dict]:
    init()
    return st.session_state.table_profiles


def table_names() -> list[str]:
    return list(get_profiles().keys())


def add_table(name: str, profile: dict) -> None:
    init()
    st.session_state.table_profiles[name] = profile


def seen_file_keys() -> set:
    init()
    return st.session_state.ingested_files


def mark_file_seen(key) -> None:
    init()
    st.session_state.ingested_files.add(key)


def get_history() -> list[dict]:
    init()
    return st.session_state.history


def recent_history(n: int = MAX_HISTORY_TURNS) -> list[dict]:
    return get_history()[-n:]


def add_history_entry(entry: dict) -> None:
    init()
    st.session_state.history.append(entry)


def reset() -> None:
    """Drop the connection and all cached state — used by the 'Start over' button."""
    con = st.session_state.get("db_con")
    if con is not None:
        try:
            con.close()
        except Exception:
            pass
    for key in ("db_con", "table_profiles", "ingested_files", "history"):
        st.session_state.pop(key, None)
    init()
