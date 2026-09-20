import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import llm_client, schema_profile


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(llm_client, "is_configured", lambda: False)
    test_app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=15).run()
    test_app.session_state["table_profiles"] = {
        "sample": schema_profile.profile_table("sample", pd.DataFrame({"id": [1]}))
    }
    yield test_app
    test_app.session_state["db_con"].close()


def show_result(app, df, **overrides):
    turn = {
        "question": "Show the result",
        "answer": "Computed result",
        "sql": "SELECT * FROM sample",
        "result_df": df,
        "truncated": False,
        "row_count": len(df),
        **overrides,
    }
    app.session_state["history"] = [turn]
    app.run()
    assert not app.exception
    return app


def test_grouped_chart_and_complete_table_render_even_with_old_cached_spec(app, monthly_region_sales):
    df = monthly_region_sales.sample(frac=1, random_state=7)
    show_result(
        app, df,
        chart_spec={"type": "line", "x": "month", "y": "monthly_sales_usd"},
    )
    charts = app.get("plotly_chart")
    assert len(charts) == 1
    plot = json.loads(charts[0].proto.spec)
    assert {trace["name"] for trace in plot["data"]} == {"East", "North", "South", "West"}
    data_expander = next(expander for expander in app.expander if expander.label == "View result data")
    assert not data_expander.proto.expanded
    assert len(data_expander.dataframe) == 1
    assert len(app.dataframe) == 1
    pd.testing.assert_frame_equal(app.dataframe[0].value, df)


def test_ambiguous_chart_shows_explanation_and_full_table(app, monthly_region_sales):
    df = pd.concat([monthly_region_sales, monthly_region_sales.iloc[[0]]], ignore_index=True)
    show_result(app, df)
    assert not app.get("plotly_chart")
    assert any("duplicate dates" in caption.value for caption in app.caption)
    assert len(app.dataframe) == 1
    pd.testing.assert_frame_equal(app.dataframe[0].value, df)


def test_all_measures_are_accessible_beneath_bar_chart(app):
    df = pd.DataFrame({
        "category": ["Electronics", "Home"],
        "sales_usd": [10000, 6000],
        "refunds_usd": [500, 200],
        "net_sales_usd": [9500, 5800],
    })
    show_result(app, df)
    assert len(app.get("plotly_chart")) == 1
    assert any(expander.label == "View result data" for expander in app.expander)
    pd.testing.assert_frame_equal(app.dataframe[0].value, df)


def test_scatter_result_also_exposes_table(app):
    df = pd.DataFrame({"a": range(20), "b": range(20)})
    show_result(app, df)
    assert len(app.get("plotly_chart")) == 1
    pd.testing.assert_frame_equal(app.dataframe[0].value, df)


def test_truncated_result_keeps_cap_and_notice(app):
    df = pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=500),
        "sales": range(500),
    })
    show_result(app, df, truncated=True, row_count=700)
    assert len(app.dataframe[0].value) == 500
    assert any(caption.value == "Showing first 500 of 700 rows." for caption in app.caption)


@pytest.mark.parametrize("df", [pd.DataFrame({"total": [42]}), pd.DataFrame({"total": []})])
def test_scalar_and_empty_results_remain_concise(app, df):
    show_result(app, df)
    assert not app.get("plotly_chart")
    assert not app.dataframe
    assert not any(expander.label == "View result data" for expander in app.expander)


def test_plain_table_is_not_duplicated(app):
    df = pd.DataFrame({"customer": ["A", "B"], "region": ["East", "West"]})
    show_result(app, df)
    assert not app.get("plotly_chart")
    assert len(app.dataframe) == 1
    pd.testing.assert_frame_equal(app.dataframe[0].value, df)


def test_describe_response_renders_without_sql_or_chart(app, monkeypatch):
    # The model itself decides whether a question needs SQL or a description -
    # there's no pre-classifier, so this question still goes to the LLM, it
    # just responds with a DESCRIBE:-prefixed answer instead of SQL.
    monkeypatch.setattr(llm_client, "is_configured", lambda: True)
    chat = Mock(return_value="DESCRIBE: The sample table holds a single id column.")
    monkeypatch.setattr(llm_client, "chat", chat)
    app.run()
    app.chat_input[0].set_value("What does this file tell about?").run()

    assert not app.exception
    assert chat.call_count == 1
    assert len(app.session_state["history"]) == 1
    entry = app.session_state["history"][0]
    assert entry["answered_from"] == "description"
    assert entry["answer"] == "The sample table holds a single id column."
    assert "sql" not in entry
    assert not app.get("plotly_chart")
    assert not app.expander


def test_chat_query_renders_grouped_trend_and_table(app, monthly_region_sales, monkeypatch):
    con = app.session_state["db_con"]
    con.register("_test_sales", monthly_region_sales)
    con.execute("CREATE TABLE monthly_sales AS SELECT * FROM _test_sales")
    con.unregister("_test_sales")
    app.session_state["table_profiles"] = {
        "monthly_sales": schema_profile.profile_table("monthly_sales", monthly_region_sales)
    }
    monkeypatch.setattr(llm_client, "is_configured", lambda: True)
    chat = Mock(side_effect=[
        'SELECT "month", "region", "monthly_sales_usd" FROM "monthly_sales" ORDER BY "month", "region"',
        "Monthly sales are shown separately for each region.",
    ])
    monkeypatch.setattr(llm_client, "chat", chat)
    app.run()
    app.chat_input[0].set_value("Show monthly sales trends separately for each region.").run()

    assert not app.exception
    assert chat.call_count == 2
    assert len(app.session_state["history"]) == 1
    pd.testing.assert_frame_equal(app.dataframe[0].value, monthly_region_sales, check_dtype=False)
    plot = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert {trace["name"] for trace in plot["data"]} == {"East", "North", "South", "West"}
