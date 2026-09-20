import pandas as pd
import pytest


@pytest.fixture
def monthly_region_sales():
    amounts = {
        "East": [5557.70, 5618.20, 3750.60],
        "North": [5191.20, 5194.00, 6972.20],
        "South": [5482.30, 6400.50, 6063.80],
        "West": [3830.20, 4364.20, 4789.70],
    }
    return pd.DataFrame(
        [
            {"month": month, "region": region, "monthly_sales_usd": values[index]}
            for index, month in enumerate(pd.date_range("2025-01-01", periods=3, freq="MS"))
            for region, values in amounts.items()
        ]
    )
