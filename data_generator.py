# =============================================================
# data_generator.py — Synthetic Data Generator
# Creates realistic demand, lot, and weather data for PISA
# Run standalone: python data_generator.py
# =============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

from config import (SKUS, FESTIVALS, WAREHOUSES, VENDORS, STORAGE_ZONES, CHILLER_MAX_IDEAL_C,
                    SENSOR_HISTORY_DAYS, SIMULATION_START, SIMULATION_END, RANDOM_SEED)

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
# HELPER: Festival multiplier for a given date
# ─────────────────────────────────────────────────────────────
def get_festival_multiplier(date, festivals):
    """Returns demand multiplier if date falls within a festival window."""
    multiplier = 1.0
    for fest in festivals:
        fest_date = datetime.strptime(fest["date"], "%Y-%m-%d").date()
        window    = fest["window_days"]
        delta     = abs((date - fest_date).days)
        if delta <= window:
            # Multiplier peaks at festival date and tapers off
            taper = 1 - (delta / (window + 1)) * 0.4
            multiplier = max(multiplier, fest["demand_multiplier"] * taper)
    return multiplier


# ─────────────────────────────────────────────────────────────
# 1. GENERATE DEMAND DATA
#    One row per (date, sku, warehouse)
#    Captures: weekly seasonality, annual seasonality,
#              festival spikes, weather effect, random noise
# ─────────────────────────────────────────────────────────────
def generate_demand_data():
    print("📊 Generating demand data...")

    date_range   = pd.date_range(start=SIMULATION_START, end=SIMULATION_END, freq="D")
    records      = []

    # Day-of-week multipliers (restaurants order less on Monday)
    DOW_MULTIPLIER = {0: 0.75, 1: 0.90, 2: 1.00, 3: 1.05,
                      4: 1.10, 5: 1.20, 6: 1.15}  # Mon=0, Sun=6

    for wh in WAREHOUSES:
        # Each warehouse has a slight scale factor
        wh_scale = np.random.uniform(0.85, 1.20)

        for sku in SKUS:
            base = sku["base_demand"] * wh_scale

            for date in date_range:
                d = date.date()

                # Weekly pattern
                dow_factor = DOW_MULTIPLIER[date.weekday()]

                # Annual seasonality: sin wave peaks in Oct-Nov (wedding season)
                day_of_year  = date.dayofyear
                annual_cycle = 1 + 0.15 * np.sin(2 * np.pi * (day_of_year - 60) / 365)

                # Festival multiplier
                fest_factor = get_festival_multiplier(d, FESTIVALS)

                # Weather effect: high temp → more cold beverages/dairy
                # (simplified: random temp between 18–42°C for Indian cities)
                temp_c = 28 + 7 * np.sin(2 * np.pi * (day_of_year - 30) / 365) + np.random.normal(0, 3)
                temp_c = np.clip(temp_c, 15, 45)

                # Meat demand drops slightly in extreme heat (> 38°C)
                heat_penalty = 0.85 if (temp_c > 38 and sku["category"] == "Meat & Poultry") else 1.0

                # Dairy demand rises slightly in heat
                heat_boost = 1.10 if (temp_c > 35 and sku["category"] == "Dairy") else 1.0

                # Combined demand with noise
                demand = (base
                          * dow_factor
                          * annual_cycle
                          * fest_factor
                          * heat_penalty
                          * heat_boost
                          * np.random.lognormal(0, 0.12))  # log-normal noise

                demand = max(0, round(demand, 1))

                records.append({
                    "date":          date.strftime("%Y-%m-%d"),
                    "sku_id":        sku["sku_id"],
                    "sku_name":      sku["name"],
                    "category":      sku["category"],
                    "warehouse_id":  wh["id"],
                    "warehouse_name":wh["name"],
                    "city":          wh["city"],
                    "actual_demand_kg": demand,
                    "price_per_kg":  sku["price_per_kg"],
                    "revenue":       round(demand * sku["price_per_kg"], 2),
                    "temp_c":        round(temp_c, 1),
                    "day_of_week":   date.day_name(),
                    "is_weekend":    int(date.weekday() >= 5),
                    "month":         date.month,
                    "day_of_year":   day_of_year,
                    "festival_multiplier": round(fest_factor, 2),
                })

    df = pd.DataFrame(records)
    df.to_csv("data/demand_data.csv", index=False)
    print(f"   ✅ Demand data: {len(df):,} rows saved → data/demand_data.csv")
    return df


# ─────────────────────────────────────────────────────────────
# 2. GENERATE LOT DATA
#    Lot = a batch of a specific SKU procured together
#    Each lot has: procurement date, quantity, expiry date,
#                  storage conditions, and a spoilage outcome
# ─────────────────────────────────────────────────────────────
def generate_lot_data():
    print("📦 Generating lot/inventory data...")

    records   = []
    lot_id    = 1000

    date_range = pd.date_range(start=SIMULATION_START, end=SIMULATION_END, freq="D")

    for wh in WAREHOUSES:
        for sku in SKUS:
            shelf_life = sku["shelf_life_days"]

            # Procurement happens every N days based on shelf life
            # (you don't order spinach weekly — you order every 2 days)
            procurement_interval = max(1, shelf_life // 2)

            proc_dates = date_range[::procurement_interval]

            sku_vendors = [v for v in VENDORS if v["category"] == sku["category"]]

            for proc_date in proc_dates:
                vendor = sku_vendors[np.random.randint(len(sku_vendors))]
                # Quantity procured: 2–4 days of base demand (slight over-ordering pattern)
                over_order_factor = np.random.uniform(1.1, 1.6)  # realistic over-ordering
                quantity_kg = round(
                    sku["base_demand"] * procurement_interval * over_order_factor
                    * np.random.lognormal(0, 0.1),
                    1
                )

                # Storage temperature: sometimes deviates (cold-chain issues)
                temp_deviation = np.random.choice(
                    [0, 0, 0, 0, 2, 4, 6],  # 4/7 chance of perfect, 3/7 chance of deviation
                    p=[0.40, 0.20, 0.15, 0.10, 0.08, 0.05, 0.02]
                )
                actual_temp = sku["ideal_temp_c"] + temp_deviation

                # Expiry date = procurement date + shelf life (adjusted for temp deviation)
                # Higher deviation → shorter effective shelf life
                shelf_life_reduction = int(temp_deviation * 0.4)
                effective_shelf_life = max(1, shelf_life - shelf_life_reduction)
                expiry_date = proc_date + timedelta(days=effective_shelf_life)

                # ── Compute age_pct at a realistic inspection point ──
                # Instead of using simulation end (which makes all lots ~1.0),
                # simulate an inspection at a random point during the lot's life
                inspection_day = np.random.randint(0, effective_shelf_life + 1)
                age_pct = min(1.0, inspection_day / effective_shelf_life)

                # Days until expiry from the inspection point
                days_to_expiry = effective_shelf_life - inspection_day

                # ── Spoilage probability: sigmoid on feature interactions ──
                # Stronger, more learnable signal for the classifier
                category_base_spoil = {
                    "Leafy Vegetables": 0.22,
                    "Other Vegetables": 0.10,
                    "Root Vegetables":  0.04,
                    "Dairy":            0.12,
                    "Meat & Poultry":   0.18,
                }
                base_spoil_prob = category_base_spoil.get(sku["category"], 0.10)

                # Sigmoid: spoilage risk accelerates sharply when lot is old AND temp is bad
                # This creates the non-linear interaction that RF can learn
                interaction_score = (
                    age_pct * 2.5                    # age is the primary driver
                    + (temp_deviation / 6.0) * 1.8   # temp deviation amplifies risk
                    + (over_order_factor - 1) * 0.6   # over-ordering adds mild risk
                    + base_spoil_prob * 1.5            # category baseline
                    + vendor["quality_reject_rate"] * 4.0  # poor inbound quality spoils faster
                )
                # Sigmoid transform: maps interaction_score → probability
                spoil_prob = 1.0 / (1.0 + np.exp(-(interaction_score - 2.0) * 2.5))
                spoil_prob = np.clip(spoil_prob, 0.02, 0.95)

                did_spoil = int(np.random.random() < spoil_prob)

                # Survival target: day the lot spoiled, or censored at end of
                # shelf life (dispatched without spoiling). Temp abuse shortens it.
                if did_spoil:
                    days_to_event = int(np.clip(round(effective_shelf_life * np.random.uniform(0.4, 1.0)
                                                      - temp_deviation * 0.3), 1, effective_shelf_life))
                else:
                    days_to_event = effective_shelf_life

                # If spoiled, how much % was spoiled?
                spoilage_pct = round(np.random.uniform(0.15, 0.80), 2) if did_spoil else 0.0
                spoiled_kg   = round(quantity_kg * spoilage_pct, 1)
                spoiled_value= round(spoiled_kg * sku["price_per_kg"], 2)

                # Spoilage risk score (0–100) — derived from the same features
                risk_score   = min(100, int(
                    (age_pct * 45)
                    + (temp_deviation * 7)
                    + (base_spoil_prob * 80)
                    + np.random.normal(0, 4)
                ))
                risk_score = max(0, risk_score)

                records.append({
                    "lot_id":           f"LOT{lot_id:04d}",
                    "sku_id":           sku["sku_id"],
                    "sku_name":         sku["name"],
                    "category":         sku["category"],
                    "warehouse_id":     wh["id"],
                    "warehouse_name":   wh["name"],
                    "vendor_id":        vendor["vendor_id"],
                    "vendor_reject_rate": vendor["quality_reject_rate"],
                    "procurement_date": proc_date.strftime("%Y-%m-%d"),
                    "expiry_date":      expiry_date.strftime("%Y-%m-%d"),
                    "shelf_life_days":  shelf_life,
                    "effective_shelf_life": effective_shelf_life,
                    "quantity_kg":      quantity_kg,
                    "over_order_factor":round(over_order_factor, 2),
                    "ideal_temp_c":     sku["ideal_temp_c"],
                    "actual_temp_c":    actual_temp,
                    "temp_deviation_c": temp_deviation,
                    "price_per_kg":     sku["price_per_kg"],
                    "lot_value":        round(quantity_kg * sku["price_per_kg"], 2),
                    "did_spoil":        did_spoil,
                    "days_to_event":    days_to_event,
                    "spoilage_pct":     spoilage_pct,
                    "spoiled_kg":       spoiled_kg,
                    "spoiled_value":    spoiled_value,
                    "days_to_expiry":   days_to_expiry,
                    "risk_score":       risk_score,
                    "age_pct":          round(age_pct, 2),
                })

                lot_id += 1

    df = pd.DataFrame(records)
    df.to_csv("data/lot_data.csv", index=False)
    print(f"   ✅ Lot data: {len(df):,} rows saved → data/lot_data.csv")
    return df


# ─────────────────────────────────────────────────────────────
# 3. SIMULATED IoT COLD-STORAGE SENSORS
#    Hourly temp/humidity per warehouse × zone for the last N days.
#    Includes door-open spikes and occasional compressor-failure episodes.
# ─────────────────────────────────────────────────────────────
def storage_zone(sku):
    return "CHILLER" if sku["ideal_temp_c"] <= CHILLER_MAX_IDEAL_C else "COOL_ROOM"


def get_ist_now():
    try:
        from zoneinfo import ZoneInfo
        return pd.Timestamp.now(ZoneInfo("Asia/Kolkata")).tz_localize(None)
    except Exception:
        import datetime
        return pd.Timestamp.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).tz_localize(None)


def generate_sensor_readings():
    print("🌡️  Generating IoT sensor readings...")
    end   = get_ist_now().floor("h")
    hours = pd.date_range(end=end, periods=SENSOR_HISTORY_DAYS * 24, freq="h")
    records = []

    for wh in WAREHOUSES:
        for zone, spec in STORAGE_ZONES.items():
            temp = spec["setpoint_c"] + np.random.normal(0, 0.4, len(hours))
            # Door-open spikes during dispatch hours (5–9 AM)
            temp += np.where(np.isin(hours.hour, [5, 6, 7, 8]), np.random.uniform(0.5, 2.0, len(hours)), 0)
            # ~1 in 3 zones has a compressor failure episode of 4–12 hours
            if np.random.random() < 0.33:
                start = np.random.randint(0, len(hours) - 12)
                temp[start:start + np.random.randint(4, 13)] += np.random.uniform(4, 8)
            humidity = np.clip(spec["humidity_pct"] + np.random.normal(0, 3, len(hours)), 50, 100)

            for ts, t, h in zip(hours, temp, humidity):
                records.append({
                    "reading_ts":   ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "warehouse_id": wh["id"],
                    "zone":         zone,
                    "sensor_id":    f"{wh['id']}_{zone}",
                    "temp_c":       round(float(t), 2),
                    "humidity_pct": round(float(h), 1),
                })

    df = pd.DataFrame(records)
    df.to_csv("data/sensor_readings.csv", index=False)
    print(f"   ✅ Sensor readings: {len(df):,} rows saved → data/sensor_readings.csv")
    return df


def generate_vendor_data():
    df = pd.DataFrame(VENDORS)
    df.to_csv("data/vendors.csv", index=False)
    print(f"   ✅ Vendors: {len(df)} rows saved → data/vendors.csv")
    return df


# ─────────────────────────────────────────────────────────────
# 4. GENERATE ACTIVE LOTS (current warehouse stock)
#    Carries the same features the spoilage model was trained on.
#    Temperature deviation comes from the last 24h of sensor data,
#    and the risk score is assigned by the model (ml_models.score_active_lots).
# ─────────────────────────────────────────────────────────────
def generate_active_lots(sensor_df=None):
    print("🚨 Generating active lots...")
    if sensor_df is None:
        sensor_df = pd.read_csv("data/sensor_readings.csv")

    # Mean zone temperature over the last 24h
    sensor_df = sensor_df.copy()
    sensor_df["reading_ts"] = pd.to_datetime(sensor_df["reading_ts"])
    recent = sensor_df[sensor_df["reading_ts"] > sensor_df["reading_ts"].max() - pd.Timedelta(hours=24)]
    zone_temp = recent.groupby(["warehouse_id", "zone"])["temp_c"].mean().to_dict()

    records = []
    today   = get_ist_now().date()

    for i, sku in enumerate(SKUS):
        shelf_life = sku["shelf_life_days"]
        wh = WAREHOUSES[i % len(WAREHOUSES)]
        sku_vendors = [v for v in VENDORS if v["category"] == sku["category"]]
        zone = storage_zone(sku)
        temp_dev = max(0.0, round(zone_temp[(wh["id"], zone)] - sku["ideal_temp_c"], 1))

        for j in range(np.random.randint(2, 5)):
            vendor         = sku_vendors[np.random.randint(len(sku_vendors))]
            days_old       = np.random.randint(0, shelf_life + 1)
            days_remaining = shelf_life - days_old
            quantity       = round(sku["base_demand"] * np.random.uniform(1.5, 3.0), 1)

            records.append({
                "lot_id":           f"LOT{2000 + i*10 + j:04d}",
                "sku_id":           sku["sku_id"],
                "sku_name":         sku["name"],
                "category":         sku["category"],
                "warehouse_id":     wh["id"],
                "warehouse_name":   wh["name"],
                "vendor_id":        vendor["vendor_id"],
                "vendor_reject_rate": vendor["quality_reject_rate"],
                "storage_zone":     zone,
                "procurement_date": (today - timedelta(days=int(days_old))).strftime("%Y-%m-%d"),
                "expiry_date":      (today + timedelta(days=int(days_remaining))).strftime("%Y-%m-%d"),
                "days_old":         int(days_old),
                "days_remaining":   int(days_remaining),
                "shelf_life_days":  shelf_life,
                "age_pct":          round(days_old / shelf_life, 2),
                "pct_shelf_remaining": round(days_remaining / shelf_life * 100, 1),
                "quantity_kg":      quantity,
                "over_order_factor": round(np.random.uniform(1.0, 1.6), 2),
                "price_per_kg":     sku["price_per_kg"],
                "lot_value_inr":    round(quantity * sku["price_per_kg"], 2),
                "temp_deviation_c": temp_dev,
            })

    df = pd.DataFrame(records)
    df.to_csv("data/active_lots.csv", index=False)
    print(f"   ✅ Active lots: {len(df)} lots saved → data/active_lots.csv")
    return df


def generate_all():
    os.makedirs("data", exist_ok=True)
    df_demand = generate_demand_data()
    df_lots   = generate_lot_data()
    generate_vendor_data()
    df_sensor = generate_sensor_readings()
    df_active = generate_active_lots(df_sensor)
    return df_demand, df_lots, df_active


# ─────────────────────────────────────────────────────────────
# MAIN — Run all generators
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🏭 Hyperpure PISA — Data Generator")
    print("=" * 45)

    df_demand, df_lots, df_active = generate_all()

    print("\n📋 Summary:")
    print(f"   Demand records : {len(df_demand):,}")
    print(f"   Historical lots: {len(df_lots):,}")
    print(f"   Active lots    : {len(df_active)}")
    print(f"   Date range     : {SIMULATION_START} → {SIMULATION_END}")
    print(f"   SKUs           : {len(SKUS)}")
    print(f"   Warehouses     : {len(WAREHOUSES)}")
    print("\n✅ All data generated successfully!\n")
