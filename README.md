# AI-Powered Data Q&A Web App

**Live App:** https://ai-powered-data-app-web-app-7uapp8grbxwcrcwbbny9cne.streamlit.app

Upload one or more CSV/Excel files and ask analytical questions about them in plain English. Answers are always computed, not guessed: the LLM only translates a question into SQL, which is then actually run against your data with DuckDB.

## Tech stack

- **Streamlit** - the app (UI and logic in one process)
- **DuckDB** (in-process, no separate database server) - runs the actual SQL queries against uploaded data
- **pandas + openpyxl** - CSV/XLSX parsing
- **sqlglot** - validates every LLM-generated query before it's allowed to run
- **OpenAI API** - via an OpenAI-compatible client
- **Plotly** - charts

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
```

Edit `.env` and set `LLM_API_KEY` to an OpenAI API key from https://platform.openai.com/api-keys.

```bash
streamlit run app.py
```

Open http://localhost:8501 and click **"Try sample HR data"** to load the bundled sample files, or upload your own CSV/XLSX files.
