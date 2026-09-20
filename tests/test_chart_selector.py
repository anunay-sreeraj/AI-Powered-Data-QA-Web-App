import pandas as pd
import pytest

from core import chart_selector


def test_single_scalar_is_text():
    df = pd.DataFrame({"total": [42]})
    assert chart_selector.select_chart_spec(df)["type"] == "text"


def test_datetime_plus_numeric_is_line():
    df = pd.DataFrame({"month": pd.to_datetime(["2024-01-01", "2024-02-01"]), "total": [10, 20]})
    spec = chart_selector.select_chart_spec(df)
    assert spec["type"] == "line"
    assert spec["x"] == "month"


def test_low_cardinality_category_plus_numeric_is_bar():
    df = pd.DataFrame({"department": ["Eng", "Sales", "Eng", "Sales"], "amount": [1, 2, 3, 4]})
    spec = chart_selector.select_chart_spec(df)
    assert spec["type"] == "bar"


def test_two_numeric_columns_with_many_rows_is_scatter():
    df = pd.DataFrame({"a": range(20), "b": range(20)})
    spec = chart_selector.select_chart_spec(df)
    assert spec["type"] == "scatter"


def test_empty_result_is_none():
    assert chart_selector.select_chart_spec(pd.DataFrame())["type"] == "none"


def test_none_result_is_none():
    assert chart_selector.select_chart_spec(None)["type"] == "none"


def test_grouped_trend_has_four_named_traces_with_correct_values(monthly_region_sales):
    df = monthly_region_sales.sample(frac=1, random_state=7)
    original = df.copy(deep=True)
    spec = chart_selector.select_chart_spec(df)
    assert spec == {"type": "line", "x": "month", "y": "monthly_sales_usd", "series": "region"}

    fig = chart_selector.build_figure(df, spec)
    assert len(fig.data) == 4
    assert {trace.name for trace in fig.data} == {"East", "North", "South", "West"}
    assert fig.layout.legend.title.text == "region"
    for trace in fig.data:
        expected = df[df["region"] == trace.name].sort_values("month")
        assert list(pd.to_datetime(trace.x)) == expected["month"].tolist()
        assert list(trace.y) == pytest.approx(expected["monthly_sales_usd"].tolist())
        assert len(trace.x) == 3
        assert trace.connectgaps is False
        assert trace.showlegend
        assert "region=" in trace.hovertemplate
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("group_name", ["category", "supplier"])
def test_grouping_is_not_hardcoded_to_region(monthly_region_sales, group_name):
    df = monthly_region_sales.rename(columns={"region": group_name})
    spec = chart_selector.select_chart_spec(df)
    assert spec["series"] == group_name
    assert len(chart_selector.build_figure(df, spec).data) == 4


def test_single_series_is_sorted_without_mutating_result():
    df = pd.DataFrame({
        "month": pd.to_datetime(["2025-03-01", "2025-01-01", "2025-02-01"]),
        "sales": [30, 10, 20],
    })
    original = df.copy(deep=True)
    spec = chart_selector.select_chart_spec(df)
    assert "series" not in spec
    fig = chart_selector.build_figure(df, spec)
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [10, 20, 30]
    assert list(pd.to_datetime(fig.data[0].x)) == sorted(df["month"].tolist())
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("omit_row", [False, True])
def test_missing_group_observation_breaks_line(monthly_region_sales, omit_row):
    df = monthly_region_sales.copy()
    missing = (df["region"] == "North") & (df["month"] == pd.Timestamp("2025-02-01"))
    if omit_row:
        df = df.loc[~missing].copy()
    else:
        df.loc[missing, "monthly_sales_usd"] = float("nan")
    original = df.copy(deep=True)
    fig = chart_selector.build_figure(df, chart_selector.select_chart_spec(df))
    trace = next(trace for trace in fig.data if trace.name == "North")
    assert len(trace.x) == 3
    assert list(pd.to_datetime(trace.x)) == list(pd.date_range("2025-01-01", periods=3, freq="MS"))
    assert pd.isna(trace.y[1])
    assert trace.y[0] == pytest.approx(5191.20)
    assert trace.y[2] == pytest.approx(6972.20)
    assert trace.connectgaps is False
    pd.testing.assert_frame_equal(df, original)


def test_single_series_null_measure_breaks_line():
    df = pd.DataFrame({
        "month": pd.date_range("2025-01-01", periods=3, freq="MS"),
        "sales": [10, None, 30],
    })
    fig = chart_selector.build_figure(df, chart_selector.select_chart_spec(df))
    assert pd.isna(fig.data[0].y[1])
    assert fig.data[0].connectgaps is False


@pytest.mark.parametrize("grouped", [False, True])
def test_duplicate_series_dates_fall_back_without_aggregation(monthly_region_sales, grouped):
    df = monthly_region_sales
    if not grouped:
        df = df.loc[df["region"] == "North"].drop(columns="region")
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    original = df.copy(deep=True)
    spec = chart_selector.select_chart_spec(df)
    assert spec["type"] == "table"
    assert "duplicate dates" in spec["reason"]
    assert chart_selector.build_figure(df, spec) is None
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize(
    ("extra_column", "reason"),
    [
        ({"warehouse": "East depot"}, "categorical columns"),
        ({"profit": 100.0}, "numeric columns"),
        ({"delivery_date": pd.Timestamp("2025-04-01")}, "date columns"),
    ],
)
def test_ambiguous_trends_fall_back_with_explanation(monthly_region_sales, extra_column, reason):
    spec = chart_selector.select_chart_spec(monthly_region_sales.assign(**extra_column))
    assert spec["type"] == "table"
    assert reason in spec["reason"]


@pytest.mark.parametrize(("column", "reason"), [("month", "missing dates"), ("region", "missing series labels")])
def test_missing_chart_keys_fall_back_with_explanation(monthly_region_sales, column, reason):
    df = monthly_region_sales.copy()
    df.loc[0, column] = None
    spec = chart_selector.select_chart_spec(df)
    assert spec["type"] == "table"
    assert reason in spec["reason"]


def test_too_many_series_fall_back_with_explanation():
    df = pd.DataFrame({
        "month": [pd.Timestamp("2025-01-01")] * 21,
        "region": [f"Region {n}" for n in range(21)],
        "sales": range(21),
    })
    spec = chart_selector.select_chart_spec(df)
    assert spec["type"] == "table"
    assert "more than 20 series" in spec["reason"]


def test_unused_categorical_labels_do_not_create_traces(monthly_region_sales):
    df = monthly_region_sales.copy()
    df["region"] = pd.Categorical(df["region"], categories=["East", "North", "South", "West", "Unused"])
    fig = chart_selector.build_figure(df, chart_selector.select_chart_spec(df))
    assert len(fig.data) == 4
    assert "Unused" not in {trace.name for trace in fig.data}


@pytest.mark.parametrize(
    ("df", "chart_type"),
    [
        (pd.DataFrame({"category": ["A", "B"], "sales": [10, 20]}), "bar"),
        (pd.DataFrame({"a": range(20), "b": range(20)}), "scatter"),
    ],
)
def test_existing_non_temporal_charts_still_render(df, chart_type):
    spec = chart_selector.select_chart_spec(df)
    assert spec["type"] == chart_type
    fig = chart_selector.build_figure(df, spec)
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == len(df)
