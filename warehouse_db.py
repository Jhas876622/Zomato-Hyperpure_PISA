# =============================================================
# warehouse_db.py — DuckDB analytical warehouse (star schema)
#
# Facts : fact_orders, fact_lots, fact_sensor_readings, fact_active_lots
# Dims  : dim_sku, dim_warehouse, dim_vendor, dim_date
# Marts : mart_* views (the "transform" layer, dbt-style SQL models)
#
# Rebuilt from the CSVs by the nightly pipeline: python warehouse_db.py
# =============================================================

import os

import duckdb
import pandas as pd

from config import SKUS, WAREHOUSES, DB_PATH, DATA_DIR

# ponytail: plain SQL views instead of a dbt project; port these to dbt models once there are >1 contributors
MART_VIEWS = {
    "mart_wastage_by_category_month": """
        SELECT s.category, date_trunc('month', l.procurement_date) AS month,
               SUM(l.lot_value)      AS procured_value_inr,
               SUM(l.spoiled_value)  AS spoiled_value_inr,
               ROUND(100 * SUM(l.spoiled_value) / SUM(l.lot_value), 2) AS wastage_pct
        FROM fact_lots l JOIN dim_sku s USING (sku_id)
        GROUP BY ALL""",
    "mart_vendor_scorecard": """
        SELECT v.vendor_id, v.name AS vendor_name, v.category,
               v.on_time_rate, v.quality_reject_rate,
               COUNT(*)                                  AS lots_supplied,
               ROUND(100 * AVG(l.did_spoil), 2)          AS spoil_rate_pct,
               SUM(l.spoiled_value)                      AS spoiled_value_inr
        FROM fact_lots l JOIN dim_vendor v USING (vendor_id)
        GROUP BY ALL""",
    "mart_daily_demand": """
        SELECT o.date, w.city, s.category,
               SUM(o.demand_kg) AS demand_kg, SUM(o.revenue_inr) AS revenue_inr,
               AVG(o.temp_c)    AS avg_temp_c
        FROM fact_orders o
        JOIN dim_sku s USING (sku_id)
        JOIN dim_warehouse w USING (warehouse_id)
        GROUP BY ALL""",
    "mart_cold_chain_excursions": """
        SELECT warehouse_id, zone, date_trunc('day', reading_ts) AS day,
               MAX(temp_c) AS max_temp_c, AVG(temp_c) AS avg_temp_c,
               COUNT(*) FILTER (WHERE temp_c > setpoint_c + 3) AS hours_above_setpoint_3c
        FROM fact_sensor_readings
        GROUP BY ALL""",
}


def build_warehouse(db_path=DB_PATH, data_dir=DATA_DIR):
    """(Re)builds the whole warehouse from CSVs atomically: write a temp file, then swap."""
    tmp_path = db_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    csv = lambda name: os.path.join(data_dir, name).replace("\\", "/")

    con = duckdb.connect(tmp_path)
    try:
        con.register("skus_df", pd.DataFrame(SKUS))
        con.register("wh_df", pd.DataFrame(WAREHOUSES))
        con.execute("""
            CREATE TABLE dim_sku AS
              SELECT sku_id, name AS sku_name, category, shelf_life_days, ideal_temp_c,
                     price_per_kg, base_demand AS base_demand_kg FROM skus_df;
            CREATE TABLE dim_warehouse AS
              SELECT id AS warehouse_id, name AS warehouse_name, city FROM wh_df;
        """)
        con.execute(f"""
            CREATE TABLE dim_vendor AS SELECT * FROM read_csv_auto('{csv("vendors.csv")}');

            CREATE TABLE fact_orders AS
              SELECT CAST(date AS DATE) AS date, sku_id, warehouse_id,
                     actual_demand_kg AS demand_kg, revenue AS revenue_inr,
                     temp_c, festival_multiplier
              FROM read_csv_auto('{csv("demand_data.csv")}');

            CREATE TABLE dim_date AS
              SELECT DISTINCT date, dayname(date) AS day_name, month(date) AS month,
                     dayofweek(date) IN (0, 6) AS is_weekend, festival_multiplier
              FROM fact_orders;

            CREATE TABLE fact_lots AS
              SELECT lot_id, sku_id, warehouse_id, vendor_id,
                     CAST(procurement_date AS DATE) AS procurement_date,
                     CAST(expiry_date AS DATE) AS expiry_date,
                     effective_shelf_life, quantity_kg, over_order_factor,
                     actual_temp_c, temp_deviation_c, lot_value, did_spoil, days_to_event,
                     spoilage_pct, spoiled_kg, spoiled_value
              FROM read_csv_auto('{csv("lot_data.csv")}');

            CREATE TABLE fact_sensor_readings AS
              SELECT CAST(reading_ts AS TIMESTAMP) AS reading_ts, warehouse_id, zone, sensor_id,
                     temp_c, humidity_pct,
                     CASE zone WHEN 'CHILLER' THEN 3.0 ELSE 12.0 END AS setpoint_c
              FROM read_csv_auto('{csv("sensor_readings.csv")}');

            CREATE TABLE fact_active_lots AS
              SELECT * FROM read_csv_auto('{csv("active_lots.csv")}');
        """)
        for name, sql in MART_VIEWS.items():
            con.execute(f"CREATE VIEW {name} AS {sql}")
    finally:
        con.close()

    os.replace(tmp_path, db_path)
    print(f"   ✅ DuckDB warehouse built → {db_path}")
    return db_path


def connect_readonly(db_path=DB_PATH):
    """Read-only connection with no filesystem/network access — safe for ad-hoc and LLM-written SQL."""
    return duckdb.connect(db_path, read_only=True, config={"enable_external_access": False})


def query(sql, db_path=DB_PATH, max_rows=200):
    con = connect_readonly(db_path)
    try:
        return con.execute(sql).fetchdf().head(max_rows)
    finally:
        con.close()


def describe_schema(db_path=DB_PATH):
    """Compact 'table(col type, ...)' listing, used to brief the analyst bot."""
    cols = query("""SELECT table_name, column_name, data_type FROM information_schema.columns
                    ORDER BY table_name, ordinal_position""", db_path, max_rows=10_000)
    return "\n".join(
        f"{t}({', '.join(f'{c} {d}' for c, d in zip(g.column_name, g.data_type))})"
        for t, g in cols.groupby("table_name", sort=True)
    )


if __name__ == "__main__":
    build_warehouse()
    print(describe_schema())
