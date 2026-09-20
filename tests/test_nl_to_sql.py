from unittest.mock import Mock

import pytest

from core import nl_to_sql, sql_guard


@pytest.fixture
def con():
    c = sql_guard.create_sandboxed_connection()
    c.execute('CREATE TABLE "employees" (employee_id VARCHAR, salary DOUBLE)')
    c.execute("INSERT INTO \"employees\" VALUES ('E1', 100.0)")
    yield c
    c.close()


def test_describe_response_is_returned_without_touching_sql_guard(con, monkeypatch):
    chat = Mock(return_value="DESCRIBE: This table tracks employee salaries by id.")
    monkeypatch.setattr(nl_to_sql.llm_client, "chat", chat)

    result = nl_to_sql.generate_and_execute("what does this file tell about?", "schema...", [], con, {"employees"})

    assert result.description == "This table tracks employee salaries by id."
    assert result.sql is None
    assert result.df is None
    assert result.error is None
    assert chat.call_count == 1  # no retry loop entered for a valid DESCRIBE response


@pytest.mark.parametrize(
    "raw_response",
    [
        "describe: lowercase prefix still counts",
        "  DESCRIBE:   extra whitespace around the prefix and text   ",
        "Describe: Mixed case prefix",
    ],
)
def test_describe_prefix_matching_is_case_insensitive_and_trims_whitespace(con, monkeypatch, raw_response):
    chat = Mock(return_value=raw_response)
    monkeypatch.setattr(nl_to_sql.llm_client, "chat", chat)

    result = nl_to_sql.generate_and_execute("a question", "schema...", [], con, {"employees"})

    assert result.description is not None
    assert not result.description.upper().startswith("DESCRIBE")
    assert result.description == result.description.strip()


def test_non_describe_response_still_goes_through_normal_sql_path(con, monkeypatch):
    chat = Mock(return_value='SELECT * FROM "employees"')
    monkeypatch.setattr(nl_to_sql.llm_client, "chat", chat)

    result = nl_to_sql.generate_and_execute("total salary?", "schema...", [], con, {"employees"})

    assert result.description is None
    assert result.error is None
    assert result.df is not None
    assert len(result.df) == 1


def test_invalid_sql_still_triggers_the_repair_loop(con, monkeypatch):
    chat = Mock(side_effect=["SELECT * FROM \"not_a_real_table\"", 'SELECT * FROM "employees"'])
    monkeypatch.setattr(nl_to_sql.llm_client, "chat", chat)

    result = nl_to_sql.generate_and_execute("total salary?", "schema...", [], con, {"employees"})

    assert chat.call_count == 2
    assert result.error is None
    assert result.attempts == 2
