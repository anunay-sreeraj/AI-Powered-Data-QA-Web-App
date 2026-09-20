import pandas as pd

from core import schema_profile


def test_sanitize_columns_strips_and_dedupes():
    df = pd.DataFrame({"  Name ": [1], "Name": [2], "a  b": [3]})
    out = schema_profile.sanitize_columns(df)
    assert list(out.columns) == ["Name", "Name_1", "a b"]


def test_coerce_types_parses_currency_strings():
    df = pd.DataFrame({"salary": ["$5,200.00", "$4,100.00", "$6,000.00"]})
    out = schema_profile.coerce_types(df)
    assert out["salary"].dtype.kind == "f"
    assert out["salary"].tolist() == [5200.0, 4100.0, 6000.0]


def test_coerce_types_leaves_ambiguous_text_alone():
    df = pd.DataFrame({"note": ["N/A", "some text", "$maybe not a number"]})
    out = schema_profile.coerce_types(df)
    assert out["note"].dtype == object


def test_coerce_types_picks_dayfirst_when_it_parses_more():
    # day=25/28 are unambiguous days, so month-first parsing must fail on them
    df = pd.DataFrame({"hire_date": ["25/03/2021", "28/01/2022", "15/06/2020"]})
    out = schema_profile.coerce_types(df)
    assert str(out["hire_date"].dtype).startswith("datetime64")
    assert out["hire_date"].notna().all()


def test_find_name_matches_detects_shared_columns_across_tables():
    profiles = {
        "employees": {"columns": [{"name": "employee_id"}, {"name": "department"}]},
        "attendance": {"columns": [{"name": "employee_id"}, {"name": "days_absent"}]},
    }
    hints = schema_profile.find_name_matches(profiles)
    assert any("employee_id" in h for h in hints)
    assert not any("days_absent" in h for h in hints)


def test_profile_table_carries_source_file_and_sheet():
    df = pd.DataFrame({"a": [1, 2]})
    profile = schema_profile.profile_table("employees", df, source_file="hr.xlsx", sheet="Employees")
    assert profile["source_file"] == "hr.xlsx"
    assert profile["sheet"] == "Employees"
