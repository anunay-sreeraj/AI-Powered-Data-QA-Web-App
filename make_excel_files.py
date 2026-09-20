import pandas as pd
from pathlib import Path

out_dir = Path(__file__).parent / "sample_data_non_hr"
out_dir.mkdir(exist_ok=True)

# 1. Retail Store Multi-sheet Excel workbook
customers_df = pd.read_csv(out_dir / "customers.csv")
products_df = pd.read_csv(out_dir / "products.csv")
orders_df = pd.read_csv(out_dir / "orders.csv")

retail_xlsx = out_dir / "retail_store.xlsx"
with pd.ExcelWriter(retail_xlsx, engine="openpyxl") as writer:
    orders_df.to_excel(writer, sheet_name="Orders", index=False)
    products_df.to_excel(writer, sheet_name="Products", index=False)
    customers_df.to_excel(writer, sheet_name="Customers", index=False)

print(f"Created {retail_xlsx}")

# 2. SaaS Metrics Multi-sheet Excel workbook
saas_df = pd.read_csv(out_dir / "saas_subscriptions.csv")
saas_xlsx = out_dir / "saas_metrics.xlsx"
with pd.ExcelWriter(saas_xlsx, engine="openpyxl") as writer:
    saas_df.to_excel(writer, sheet_name="Subscriptions", index=False)

print(f"Created {saas_xlsx}")
