"""Streamlit entrypoint: upload CSV/Excel files, ask questions in plain English,
get back an answer that was actually computed (via LLM-generated, validated,
sandboxed SQL against DuckDB) rather than guessed by the model - plus a chart
where the result shape calls for one.
"""
from pathlib import Path

import streamlit as st

from core import chart_selector, ingestion, llm_client, nl_to_sql, schema_profile, session_store


class _LocalSampleFile:
    """Adapts a local file path to the minimal interface ingestion.py expects
    (the same subset of Streamlit's UploadedFile: .name, .size, .seek, .read),
    so the 'try sample data' button can reuse the exact same ingestion path
    as a real upload."""

    def __init__(self, path: Path):
        self.name = path.name
        self.size = path.stat().st_size
        self._fh = open(path, "rb")

    def seek(self, *args, **kwargs):
        return self._fh.seek(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self._fh.read(*args, **kwargs)

st.set_page_config(page_title="Data Q&A Agent", page_icon="📊", layout="wide")
session_store.init()

st.title("📊 Data Q&A Agent")
st.caption(
    "Upload CSV/Excel files and ask analytical questions in plain English. "
    "The model only ever translates your question into SQL - every answer is computed "
    "by actually running that query against your data, never guessed."
)

with st.sidebar:
    st.subheader("Session")
    if st.button("Start over", use_container_width=True):
        session_store.reset()
        st.rerun()
    st.divider()
    st.caption(
        f"Limits: {ingestion.MAX_FILE_SIZE_BYTES // (1024*1024)}MB per file · "
        f"500 rows per result · 15s query timeout"
    )

if not llm_client.is_configured():
    st.warning(
        "**LLM not configured.** Copy `.env.example` to `.env` and add an API key "
        "(OpenAI / Groq / Ollama - see the file for details), then restart the app. "
        "You can still upload files and explore the schema summaries below without it.",
        icon="⚠️",
    )

upload_col, sample_col = st.columns([3, 1])
with upload_col:
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
    )
with sample_col:
    st.write("")
    st.write("")
    load_sample = st.button("Try sample HR data", use_container_width=True)

seen = session_store.seen_file_keys()
new_files = []
if uploaded_files:
    new_files.extend(f for f in uploaded_files if (f.name, f.size) not in seen)

if load_sample:
    sample_dir = Path(__file__).parent / "sample_data"
    new_files.extend(
        _LocalSampleFile(path)
        for path in sorted(sample_dir.glob("*.csv"))
        if (path.name, path.stat().st_size) not in seen
    )

if new_files:
    with st.spinner(f"Ingesting {len(new_files)} file(s)..."):
        con = session_store.get_connection()
        report = ingestion.ingest_files(new_files, con, existing_tables=set(session_store.table_names()))
        for result in report.results:
            df = con.execute(f'SELECT * FROM "{result.table_name}"').fetch_df()
            profile = schema_profile.profile_table(
                result.table_name, df, source_file=result.source_file, sheet=result.sheet
            )
            session_store.add_table(result.table_name, profile)
        for f in new_files:
            session_store.mark_file_seen((f.name, f.size))

    for w in report.warnings:
        st.warning(w)
    if report.results:
        st.success(f"Loaded {len(report.results)} table(s): " + ", ".join(r.table_name for r in report.results))

profiles = session_store.get_profiles()

if not profiles:
    st.info("Upload one or more CSV/Excel files to get started.")
    st.stop()

st.subheader("Uploaded tables")
cols = st.columns(min(len(profiles), 3))
for i, (name, profile) in enumerate(profiles.items()):
    with cols[i % len(cols)]:
        with st.container(border=True):
            st.markdown(f"**{name}** · {profile['row_count']} rows")
            st.caption(", ".join(c["name"] for c in profile["columns"]))

st.divider()
st.subheader("Ask a question")
st.caption(
    'Try: "What is the total monthly salary cost by department?", '
    '"Compare average days absent between departments", or "Show the attendance trend over time."'
)


def render_turn(turn: dict, idx: int) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("error"):
            st.error(turn["error"])
            if turn.get("sql"):
                with st.expander("Last attempted SQL"):
                    st.code(turn["sql"], language="sql")
            return

        st.write(turn["answer"])
        if turn.get("answered_from") == "description":
            st.caption("Described from your files' schema - no data query was run.")
        if turn.get("sql"):
            with st.expander("View generated SQL"):
                st.code(turn["sql"], language="sql")

        df = turn.get("result_df")
        if df is not None:
            spec = chart_selector.select_chart_spec(df)
            if spec.get("type") in ("line", "bar", "scatter"):
                fig = chart_selector.build_figure(df, spec)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")
                with st.expander("View result data"):
                    st.dataframe(df, use_container_width=True, key=f"df_{idx}")
            elif spec.get("type") == "table":
                if spec.get("reason"):
                    st.caption(spec["reason"])
                st.dataframe(df, use_container_width=True, key=f"df_{idx}")
            if turn.get("truncated"):
                st.caption(f"Showing first {len(df)} of {turn.get('row_count')} rows.")


for i, turn in enumerate(session_store.get_history()):
    render_turn(turn, i)

question = st.chat_input("Ask a question about your data...")
if question:
    entry: dict = {"question": question}

    if not llm_client.is_configured():
        entry["error"] = "LLM is not configured. Add an API key to .env first (see the warning above)."
    else:
        with st.spinner("Thinking..."):
            try:
                con = session_store.get_connection()
                context = schema_profile.format_schema_context(profiles, schema_profile.find_name_matches(profiles))
                history = session_store.recent_history()
                allowed = set(profiles.keys())
                result = nl_to_sql.generate_and_execute(question, context, history, con, allowed)
            except Exception as e:
                result = None
                entry["error"] = f"Request to the LLM failed: {e}"

        if result is not None:
            if result.description is not None:
                entry["answer"] = result.description
                entry["answered_from"] = "description"
            else:
                entry["sql"] = result.sql
                if result.error:
                    entry["error"] = f"Couldn't get a valid answer after {result.attempts} attempt(s). Last error: {result.error}"
                else:
                    entry["answer"] = nl_to_sql.phrase_answer(question, result.df)
                    entry["result_df"] = result.df
                    entry["truncated"] = result.truncated
                    entry["row_count"] = result.row_count

    session_store.add_history_entry(entry)
    st.rerun()
