"""CSV/XLSX upload -> sanitized, type-coerced DuckDB tables.

Each Excel sheet becomes its own table. Every file/sheet is parsed inside
its own try/except so one malformed file or sheet never fails the rest of
the batch — the report simply lists what loaded and what didn't.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from core import schema_profile

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
SUPPORTED_EXTENSIONS = (".csv", ".xlsx")


@dataclass
class IngestResult:
    table_name: str
    source_file: str
    sheet: str | None
    row_count: int
    col_count: int


@dataclass
class IngestReport:
    results: list[IngestResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _safe_table_name(base: str, existing: set[str]) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]", "_", base).strip("_").lower()
    if not name:
        name = "table"
    if re.match(r"^\d", name):
        name = f"t_{name}"
    candidate = name
    i = 2
    while candidate in existing:
        candidate = f"{name}_{i}"
        i += 1
    return candidate


def _read_csv(file) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError as e:
            last_error = e
            continue
    raise last_error or ValueError("Could not decode CSV file.")


def _read_excel_sheets(file) -> dict[str, pd.DataFrame]:
    file.seek(0)
    return pd.read_excel(file, sheet_name=None, engine="openpyxl")


def ingest_files(
    files: list,
    con: duckdb.DuckDBPyConnection,
    existing_tables: set[str],
) -> IngestReport:
    report = IngestReport()
    existing = set(existing_tables)

    for file in files:
        size = getattr(file, "size", None)
        if size is not None and size > MAX_FILE_SIZE_BYTES:
            limit_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            report.warnings.append(f'Skipped "{file.name}": exceeds the {limit_mb}MB upload limit.')
            continue

        base_name = Path(file.name).stem
        ext = Path(file.name).suffix.lower()

        try:
            if ext == ".csv":
                sheets: dict[str | None, pd.DataFrame] = {None: _read_csv(file)}
            elif ext == ".xlsx":
                sheets = _read_excel_sheets(file)
            else:
                report.warnings.append(f'Skipped "{file.name}": unsupported file type "{ext or "(none)"}".')
                continue
        except Exception as e:
            report.warnings.append(f'Skipped "{file.name}": could not parse file ({e}).')
            continue

        multi_sheet = len(sheets) > 1
        for sheet_name, df in sheets.items():
            label = file.name if sheet_name is None else f'"{sheet_name}" sheet of "{file.name}"'
            try:
                if df is None or df.shape[1] == 0 or df.dropna(how="all").empty:
                    report.warnings.append(f"Skipped {label}: no data found.")
                    continue

                table_label = base_name if not multi_sheet else f"{base_name}_{sheet_name}"
                table_name = _safe_table_name(table_label, existing)
                existing.add(table_name)

                clean_df = schema_profile.sanitize_columns(df)
                clean_df = schema_profile.coerce_types(clean_df)

                con.register("_tmp_ingest_view", clean_df)
                try:
                    con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _tmp_ingest_view')
                finally:
                    con.unregister("_tmp_ingest_view")

                report.results.append(
                    IngestResult(
                        table_name=table_name,
                        source_file=file.name,
                        sheet=sheet_name,
                        row_count=len(clean_df),
                        col_count=len(clean_df.columns),
                    )
                )
            except Exception as e:
                report.warnings.append(f"Failed to load {label}: {e}")
                continue

    return report
