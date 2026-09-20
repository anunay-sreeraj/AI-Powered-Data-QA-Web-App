"""Column sanitization, best-effort type coercion, and schema profiling.

Keeps LLM prompts small and grounded: instead of ever sending raw file
contents to the model, we send a compact per-table profile (columns, types,
null rates, a few sample rows, low-cardinality distinct values) plus a cheap
cross-table name-match hint. This is computed once per upload and reused for
every question in the session.
"""
from __future__ import annotations

import re

import pandas as pd

_CURRENCY_RE = re.compile(r"[,\$€£₹]")
_PAREN_NEGATIVE_RE = re.compile(r"^\((.*)\)$")
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _clean_column_name(name: object) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name if name else "column"


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip/collapse whitespace in column names and de-duplicate collisions."""
    df = df.copy()
    cleaned = [_clean_column_name(c) for c in df.columns]
    seen: dict[str, int] = {}
    result = []
    for name in cleaned:
        key = name.lower()
        if key in seen:
            seen[key] += 1
            result.append(f"{name}_{seen[key]}")
        else:
            seen[key] = 0
            result.append(name)
    df.columns = result
    return df


def _try_numeric(series: pd.Series) -> pd.Series | None:
    if series.dtype != object:
        return None
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return None
    stripped = non_null.str.replace(_CURRENCY_RE, "", regex=True)
    stripped = stripped.str.replace(_PAREN_NEGATIVE_RE, r"-\1", regex=True)
    is_numeric = stripped.str.match(_NUMERIC_RE)
    if is_numeric.mean() < 0.9:
        return None
    converted = pd.to_numeric(stripped, errors="coerce")
    out = pd.Series(index=series.index, dtype="float64")
    out.loc[non_null.index] = converted.to_numpy()
    return out


def _try_datetime(series: pd.Series) -> pd.Series | None:
    if series.dtype != object:
        return None
    non_null = series.dropna()
    if len(non_null) < 3:
        return None
    default = pd.to_datetime(series, errors="coerce", format="mixed")
    dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
    default_ok = default.notna().mean()
    dayfirst_ok = dayfirst.notna().mean()
    best, best_ok = (default, default_ok) if default_ok >= dayfirst_ok else (dayfirst, dayfirst_ok)
    if best_ok < 0.7:
        return None
    return best


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort numeric/date coercion for object columns.

    Only converts a column when a high fraction of its values parse cleanly;
    otherwise leaves it as text rather than guessing.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
            continue
        numeric = _try_numeric(df[col])
        if numeric is not None:
            df[col] = numeric
            continue
        dt = _try_datetime(df[col])
        if dt is not None:
            df[col] = dt
    return df


def profile_table(
    name: str,
    df: pd.DataFrame,
    max_samples: int = 3,
    low_cardinality: int = 15,
    source_file: str | None = None,
    sheet: str | None = None,
) -> dict:
    columns = []
    for col in df.columns:
        series = df[col]
        col_info: dict = {
            "name": col,
            "dtype": str(series.dtype),
            "null_pct": round(float(series.isna().mean()) * 100, 1),
        }
        nunique = series.nunique(dropna=True)
        if 0 < nunique <= low_cardinality:
            col_info["distinct_values"] = [str(v) for v in series.dropna().unique().tolist()]
        columns.append(col_info)
    sample_rows = df.head(max_samples).astype(str).to_dict(orient="records")
    return {
        "table": name,
        "row_count": int(len(df)),
        "columns": columns,
        "sample_rows": sample_rows,
        "source_file": source_file,
        "sheet": sheet,
    }


def profile_all(tables: dict[str, pd.DataFrame]) -> dict[str, dict]:
    return {name: profile_table(name, df) for name, df in tables.items()}


def find_name_matches(profiles: dict[str, dict]) -> list[str]:
    """Cheap cross-table hint: exact (case/whitespace-insensitive) column name matches.

    Deliberately not a scored join-key-detection subsystem (false positives on
    unrelated low-cardinality columns, false negatives on real-world drift like
    EmpID vs Employee_Code, threshold tuning eats hours) — this is a one-line
    hint. The LLM gets full schema + sample values for every table and writes
    the actual JOIN itself.
    """
    by_norm: dict[str, list[str]] = {}
    for table, profile in profiles.items():
        for col in profile["columns"]:
            norm = col["name"].strip().lower()
            by_norm.setdefault(norm, []).append(f'{table}."{col["name"]}"')
    hints = []
    for refs in by_norm.values():
        tables_involved = {r.split(".")[0] for r in refs}
        if len(tables_involved) > 1:
            hints.append(" ~= ".join(refs))
    return hints


def format_schema_context(profiles: dict[str, dict], hints: list[str]) -> str:
    """Render the full schema context block inserted into every LLM prompt."""
    lines = []
    for table, profile in profiles.items():
        lines.append(f'Table "{table}" ({profile["row_count"]} rows):')
        for col in profile["columns"]:
            extra = f' distinct values: {col["distinct_values"]}' if "distinct_values" in col else ""
            lines.append(f'  - "{col["name"]}" ({col["dtype"]}, {col["null_pct"]}% null){extra}')
        if profile["sample_rows"]:
            lines.append(f"  sample rows: {profile['sample_rows']}")
        lines.append("")
    if hints:
        lines.append("Possible join keys (same column name across tables, verify before relying on it):")
        for h in hints:
            lines.append(f"  - {h}")
    return "\n".join(lines)
