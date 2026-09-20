"""The SQL guard is the safety boundary around executing LLM-generated SQL -
these cases are the concrete "why this is safe" story for the panel."""
import duckdb
import pytest

from core import sql_guard

ALLOWED = {"employees", "attendance"}


@pytest.fixture
def con():
    c = sql_guard.create_sandboxed_connection()
    c.execute('CREATE TABLE "employees" (employee_id VARCHAR, salary DOUBLE)')
    c.execute('CREATE TABLE "attendance" (employee_id VARCHAR, days_absent INTEGER)')
    yield c
    c.close()


@pytest.mark.parametrize(
    "sql",
    [
        'SELECT "salary" FROM "employees"',
        'SELECT e."employee_id", SUM(a."days_absent") FROM "employees" e '
        'JOIN "attendance" a ON e."employee_id" = a."employee_id" GROUP BY 1',
        'WITH t AS (SELECT * FROM "employees") SELECT * FROM t',
        'SELECT * FROM "employees" UNION SELECT * FROM "employees"',
    ],
)
def test_allows_legitimate_read_only_queries(sql):
    result = sql_guard.validate_sql(sql, ALLOWED)
    assert result.ok, result.error


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SELECT * FROM read_parquet('s3://bucket/x.parquet')",
        "SELECT * FROM sqlite_scan('x.db', 'orders')",
        'SELECT * FROM "employees"; DROP TABLE "employees";',
        'DROP TABLE "employees"',
        "ATTACH 'x.db' AS x",
        "PRAGMA database_list",
        'COPY "employees" TO \'out.csv\'',
        'SELECT * FROM "not_a_real_table"',
        "SELECT * FROM (SELECT * FROM read_csv_auto('x')) t",
    ],
)
def test_rejects_unsafe_or_unknown_queries(sql):
    result = sql_guard.validate_sql(sql, ALLOWED)
    assert not result.ok


def test_execute_returns_dataframe(con):
    con.execute('INSERT INTO "employees" VALUES (\'E1\', 100.0)')
    result = sql_guard.execute_sql(con, 'SELECT * FROM "employees"')
    assert result.error is None
    assert len(result.df) == 1


def test_execute_reports_error_without_raising(con):
    result = sql_guard.execute_sql(con, 'SELECT "not_a_column" FROM "employees"')
    assert result.df is None
    assert result.error is not None


def test_row_cap_truncates(con):
    con.execute('INSERT INTO "employees" SELECT CAST(i AS VARCHAR), i FROM range(10) t(i)')
    result = sql_guard.execute_sql(con, 'SELECT * FROM "employees"', row_cap=3)
    assert result.truncated is True
    assert len(result.df) == 3
    assert result.row_count == 10


def test_timeout_interrupts_runaway_query_and_connection_survives(con):
    result = sql_guard.execute_sql(con, "SELECT COUNT(*) FROM range(200000000) a, range(1000) b", timeout_s=1)
    assert result.error is not None
    # connection must still be usable after an interrupt
    follow_up = sql_guard.execute_sql(con, 'SELECT COUNT(*) FROM "employees"')
    assert follow_up.error is None
