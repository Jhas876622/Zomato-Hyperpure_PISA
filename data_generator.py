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

from config import SKUS, FESTIVALS, WAREHOUSES, SIMULATION_START, SIMULATION_END, RANDOM_SEED

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

            for proc_date in proc_dates:
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
                )
                # Sigmoid transform: maps interaction_score → probability
                spoil_prob = 1.0 / (1.0 + np.exp(-(interaction_score - 2.0) * 2.5))
                spoil_prob = np.clip(spoil_prob, 0.02, 0.95)

                did_spoil = int(np.random.random() < spoil_prob)

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
# 3. GENERATE ACTIVE LOTS (for live dashboard demo)
#    These are "current" lots in the warehouse right now
#    with varying risk levels for the alert system
# ─────────────────────────────────────────────────────────────
def generate_active_lots():
    print("🚨 Generating active lots for live alerts...")

    records  = []
    today    = datetime.now().date()

    actions = {
        "CRITICAL": "🔴 Redistribute or discount immediately (< 12h)",
        "HIGH":     "🟠 Offer 25% discount to bulk buyers today",
        "MEDIUM":   "🟡 Prioritise dispatch in next shipment",
        "LOW":      "🟢 Monitor — on track for normal dispatch",
    }

    for i, sku in enumerate(SKUS):
        shelf_life = sku["shelf_life_days"]
        wh = WAREHOUSES[i % len(WAREHOUSES)]

        # Create 3-4 lots per SKU with varying ages
        for j in range(np.random.randint(2, 5)):
            days_old       = np.random.randint(0, shelf_life + 1)
            days_remaining = shelf_life - days_old
            pct_remaining  = days_remaining / shelf_life

            proc_date   = today - timedelta(days=days_old)
            expiry_date = today + timedelta(days=days_remaining)

            # Assign risk level
            if pct_remaining < 0.20:
                risk_level = "CRITICAL"
                risk_score = np.random.randint(82, 100)
            elif pct_remaining < 0.40:
                risk_level = "HIGH"
                risk_score = np.random.randint(62, 82)
            elif pct_remaining < 0.60:
                risk_level = "MEDIUM"
                risk_score = np.random.randint(40, 62)
            else:
                risk_level = "LOW"
                risk_score = np.random.randint(10, 40)

            quantity = round(sku["base_demand"] * np.random.uniform(1.5, 3.0), 1)

            records.append({
                "lot_id":           f"LOT{2000 + i*10 + j:04d}",
                "sku_name":         sku["name"],
                "category":         sku["category"],
                "warehouse_name":   wh["name"],
                "procurement_date": proc_date.strftime("%Y-%m-%d"),
                "expiry_date":      expiry_date.strftime("%Y-%m-%d"),
                "days_remaining":   days_remaining,
                "shelf_life_days":  shelf_life,
                "pct_shelf_remaining": round(pct_remaining * 100, 1),
                "quantity_kg":      quantity,
                "price_per_kg":     sku["price_per_kg"],
                "lot_value_inr":    round(quantity * sku["price_per_kg"], 2),
                "risk_score":       risk_score,
                "risk_level":       risk_level,
                "recommended_action": actions[risk_level],
                "temp_deviation_c": np.random.choice([0, 0, 0, 2, 4]),
            })

    df = pd.DataFrame(records)
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    df.to_csv("data/active_lots.csv", index=False)
    print(f"   ✅ Active lots: {len(df)} lots saved → data/active_lots.csv")
    return df


# ─────────────────────────────────────────────────────────────
# MAIN — Run all generators
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("\n🏭 Hyperpure PISA — Data Generator")
    print("=" * 45)

    df_demand = generate_demand_data()
    df_lots   = generate_lot_data()
    df_active = generate_active_lots()

    print("\n📋 Summary:")
    print(f"   Demand records : {len(df_demand):,}")
    print(f"   Historical lots: {len(df_lots):,}")
    print(f"   Active lots    : {len(df_active)}")
    print(f"   Date range     : {SIMULATION_START} → {SIMULATION_END}")
    print(f"   SKUs           : {len(SKUS)}")
    print(f"   Warehouses     : {len(WAREHOUSES)}")
    print("\n✅ All data generated successfully!\n")
