# =============================================================
# config.py — Hyperpure PISA Project Configuration
# All constants, SKU definitions, festival calendar go here
# =============================================================

# ── Zomato Brand Color ────────────────────────────────────────
ZOMATO_RED   = "#CB202D"
ZOMATO_DARK  = "#1A1A1A"
ZOMATO_GREY  = "#5A5A5A"
ZOMATO_LIGHT = "#FBE9EA"

# ── SKU Master Data ───────────────────────────────────────────
# shelf_life_days = how many days the item stays fresh in ideal storage
# ideal_temp_c    = ideal storage temperature in Celsius
# base_daily_demand_kg = average kg sold per day per warehouse

SKUS = [
    # Leafy Vegetables (fastest to spoil)
    {"sku_id": "SKU001", "name": "Spinach",       "category": "Leafy Vegetables",  "shelf_life_days": 3,  "ideal_temp_c": 4,  "price_per_kg": 45,  "base_demand": 28},
    {"sku_id": "SKU002", "name": "Coriander",     "category": "Leafy Vegetables",  "shelf_life_days": 4,  "ideal_temp_c": 4,  "price_per_kg": 80,  "base_demand": 18},
    {"sku_id": "SKU003", "name": "Fenugreek",     "category": "Leafy Vegetables",  "shelf_life_days": 3,  "ideal_temp_c": 4,  "price_per_kg": 60,  "base_demand": 12},
    {"sku_id": "SKU004", "name": "Mint Leaves",   "category": "Leafy Vegetables",  "shelf_life_days": 4,  "ideal_temp_c": 4,  "price_per_kg": 90,  "base_demand": 10},

    # Other Vegetables
    {"sku_id": "SKU005", "name": "Tomato",        "category": "Other Vegetables",  "shelf_life_days": 7,  "ideal_temp_c": 12, "price_per_kg": 35,  "base_demand": 55},
    {"sku_id": "SKU006", "name": "Capsicum",      "category": "Other Vegetables",  "shelf_life_days": 8,  "ideal_temp_c": 10, "price_per_kg": 65,  "base_demand": 30},
    {"sku_id": "SKU007", "name": "Mushroom",      "category": "Other Vegetables",  "shelf_life_days": 5,  "ideal_temp_c": 4,  "price_per_kg": 180, "base_demand": 20},
    {"sku_id": "SKU008", "name": "Brinjal",       "category": "Other Vegetables",  "shelf_life_days": 7,  "ideal_temp_c": 12, "price_per_kg": 30,  "base_demand": 22},

    # Root Vegetables (longest shelf life)
    {"sku_id": "SKU009", "name": "Onion",         "category": "Root Vegetables",   "shelf_life_days": 14, "ideal_temp_c": 15, "price_per_kg": 28,  "base_demand": 70},
    {"sku_id": "SKU010", "name": "Potato",        "category": "Root Vegetables",   "shelf_life_days": 14, "ideal_temp_c": 12, "price_per_kg": 22,  "base_demand": 80},
    {"sku_id": "SKU011", "name": "Ginger",        "category": "Root Vegetables",   "shelf_life_days": 12, "ideal_temp_c": 12, "price_per_kg": 95,  "base_demand": 15},
    {"sku_id": "SKU012", "name": "Garlic",        "category": "Root Vegetables",   "shelf_life_days": 15, "ideal_temp_c": 15, "price_per_kg": 110, "base_demand": 14},

    # Dairy
    {"sku_id": "SKU013", "name": "Paneer 200g",   "category": "Dairy",             "shelf_life_days": 7,  "ideal_temp_c": 4,  "price_per_kg": 340, "base_demand": 35},
    {"sku_id": "SKU014", "name": "Curd 500g",     "category": "Dairy",             "shelf_life_days": 5,  "ideal_temp_c": 4,  "price_per_kg": 80,  "base_demand": 25},
    {"sku_id": "SKU015", "name": "Milk Full Cream","category": "Dairy",            "shelf_life_days": 3,  "ideal_temp_c": 4,  "price_per_kg": 60,  "base_demand": 40},
    {"sku_id": "SKU016", "name": "Cheese Slice",  "category": "Dairy",             "shelf_life_days": 10, "ideal_temp_c": 4,  "price_per_kg": 450, "base_demand": 15},

    # Meat & Poultry
    {"sku_id": "SKU017", "name": "Chicken Breast","category": "Meat & Poultry",    "shelf_life_days": 3,  "ideal_temp_c": 2,  "price_per_kg": 220, "base_demand": 45},
    {"sku_id": "SKU018", "name": "Mutton Curry",  "category": "Meat & Poultry",    "shelf_life_days": 3,  "ideal_temp_c": 2,  "price_per_kg": 680, "base_demand": 22},
    {"sku_id": "SKU019", "name": "Chicken Mince", "category": "Meat & Poultry",    "shelf_life_days": 2,  "ideal_temp_c": 2,  "price_per_kg": 200, "base_demand": 20},
    {"sku_id": "SKU020", "name": "Fish Fillet",   "category": "Meat & Poultry",    "shelf_life_days": 2,  "ideal_temp_c": 0,  "price_per_kg": 320, "base_demand": 18},
]

# ── Festival Calendar (affects demand significantly) ──────────
FESTIVALS = [
    {"name": "Pongal / Makar Sankranti", "date": "2024-01-15", "demand_multiplier": 1.6, "window_days": 3},
    {"name": "Republic Day",             "date": "2024-01-26", "demand_multiplier": 1.3, "window_days": 2},
    {"name": "Maha Shivratri",           "date": "2024-03-08", "demand_multiplier": 1.5, "window_days": 2},
    {"name": "Holi",                     "date": "2024-03-25", "demand_multiplier": 2.0, "window_days": 3},
    {"name": "Ram Navami",               "date": "2024-04-17", "demand_multiplier": 1.4, "window_days": 2},
    {"name": "Eid ul-Fitr",             "date": "2024-04-10", "demand_multiplier": 2.2, "window_days": 4},
    {"name": "Independence Day",         "date": "2024-08-15", "demand_multiplier": 1.4, "window_days": 2},
    {"name": "Onam",                     "date": "2024-09-15", "demand_multiplier": 1.7, "window_days": 5},
    {"name": "Navratri / Durga Puja",   "date": "2024-10-03", "demand_multiplier": 1.9, "window_days": 9},
    {"name": "Diwali",                   "date": "2024-11-01", "demand_multiplier": 2.5, "window_days": 5},
    {"name": "Eid ul-Adha",             "date": "2024-06-17", "demand_multiplier": 2.0, "window_days": 3},
    {"name": "Christmas",                "date": "2024-12-25", "demand_multiplier": 1.8, "window_days": 4},
    {"name": "New Year Eve",             "date": "2024-12-31", "demand_multiplier": 2.0, "window_days": 2},
]

# ── Warehouse Config ──────────────────────────────────────────
WAREHOUSES = [
    {"id": "WH_DEL_01", "name": "Delhi North Hub",    "city": "Delhi"},
    {"id": "WH_MUM_01", "name": "Mumbai Andheri Hub", "city": "Mumbai"},
    {"id": "WH_BLR_01", "name": "Bangalore Hebbal",   "city": "Bangalore"},
]

# ── Model & Simulation Settings ───────────────────────────────
SIMULATION_START = "2024-01-01"
SIMULATION_END   = "2024-12-31"
FORECAST_HORIZON = 14   # days to forecast ahead
RANDOM_SEED      = 42

# Spoilage thresholds (% of shelf life remaining)
ALERT_RED    = 0.20   # < 20% shelf life left → CRITICAL
ALERT_ORANGE = 0.40   # 20-40% shelf life left → WARNING
ALERT_YELLOW = 0.60   # 40-60% shelf life left → CAUTION
# > 60% → SAFE (GREEN)

# Cost assumptions for Newsvendor Model (for interview discussion)
COST_OF_OVERSTOCKING_PCT  = 1.00   # 100% loss if spoiled
COST_OF_UNDERSTOCKING_PCT = 0.25   # 25% margin + trust cost if stockout
