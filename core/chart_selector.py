"""Deterministic, rule-based chart-type selection from a SQL result's shape.

No extra LLM call for this - keeps latency low and the behavior predictable
and easy to explain, unlike asking a model to "pick a chart type" too.
"""
from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

LOW_CARDINALITY = 20


class ChartSpec(TypedDict):
    type: Literal["none", "text", "table", "line", "bar", "scatter"]
    x: NotRequired[str]
    y: NotRequired[str]
    series: NotRequired[str]
    reason: NotRequired[str]


def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _table_spec(reason: str) -> ChartSpec:
    return {"type": "table", "reason": f"Showing result data instead of a trend chart: {reason}"}


def _select_line_spec(
    df: pd.DataFrame, datetime_cols: list[str], numeric_cols: list[str], text_cols: list[str]
) -> ChartSpec:
    if len(datetime_cols) != 1:
        return _table_spec("multiple date columns make the time axis ambiguous.")
    if len(numeric_cols) != 1:
        return _table_spec("multiple numeric columns were returned. Select one measure for a trend chart.")
    if len(text_cols) > 1:
        return _table_spec("multiple categorical columns make the series grouping ambiguous.")

    x, y = datetime_cols[0], numeric_cols[0]
    if df[x].isna().any():
        return _table_spec("some rows have missing dates.")

    spec: ChartSpec = {"type": "line", "x": x, "y": y}
    keys = [x]
    if text_cols:
        series = text_cols[0]
        if df[series].isna().any():
            return _table_spec("some rows have missing series labels.")
        if df[series].nunique() > LOW_CARDINALITY:
            return _table_spec(f"more than {LOW_CARDINALITY} series would make the chart difficult to read.")
        spec["series"] = series
        keys.append(series)

    if df.duplicated(subset=keys).any():
        return _table_spec("duplicate dates within a series need explicit aggregation in the query.")
    return spec


def select_chart_spec(df: pd.DataFrame | None) -> ChartSpec:
    if df is None or df.empty:
        return {"type": "none"}

    rows, cols = df.shape
    if rows == 1 and cols == 1:
        return {"type": "text"}

    columns = list(df.columns)
    datetime_cols = [c for c in columns if _is_datetime(df[c])]
    numeric_cols = [c for c in columns if _is_numeric(df[c])]
    text_cols = [c for c in columns if c not in datetime_cols and c not in numeric_cols]

    if datetime_cols and numeric_cols:
        return _select_line_spec(df, datetime_cols, numeric_cols, text_cols)

    if text_cols and numeric_cols:
        cat_col = text_cols[0]
        if 1 < df[cat_col].nunique(dropna=True) <= LOW_CARDINALITY and rows <= 200:
            return {"type": "bar", "x": cat_col, "y": numeric_cols[0]}

    if len(numeric_cols) >= 2 and rows > 15:
        return {"type": "scatter", "x": numeric_cols[0], "y": numeric_cols[1]}

    return {"type": "table"}


def build_figure(df: pd.DataFrame, spec: ChartSpec) -> go.Figure | None:
    chart_type = spec.get("type")
    if chart_type == "line":
        x, y = spec["x"], spec["y"]
        series = spec.get("series")
        if series:
            dates = pd.Index(df[x].drop_duplicates().sort_values(), name=x)
            groups = []
            for label, group in df.groupby(series, sort=True, observed=True):
                # A missing observation at another series' date must break the line.
                points = group[[x, y]].set_index(x).reindex(dates).reset_index()
                points[series] = label
                groups.append(points)
            plot_df = pd.concat(groups, ignore_index=True)
        else:
            plot_df = df.sort_values(x)
        fig = px.line(plot_df, x=x, y=y, color=series, markers=True)
        fig.update_traces(connectgaps=False)
        return fig
    if chart_type == "bar":
        return px.bar(df, x=spec["x"], y=spec["y"])
    if chart_type == "scatter":
        return px.scatter(df, x=spec["x"], y=spec["y"])
    return None
