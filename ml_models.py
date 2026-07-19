# =============================================================
# ml_models.py — ML Model Training for PISA
#
# Model 1: Spoilage Risk Classifier (Random Forest)
#   → Predicts: Will this lot spoil? (+ risk score 0-100)
#   → Features: age, temp deviation, category, quantity, etc.
#
# Model 2: Demand Forecasting (Rolling + Trend)
#   → Predicts: How much of SKU X will be ordered tomorrow?
#   → Uses: lag features, day-of-week, festivals, weather
#
# Run standalone: python ml_models.py
# =============================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_percentage_error, f1_score)

from config import SKUS, RANDOM_SEED

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
# MODEL 1: SPOILAGE RISK CLASSIFIER
# ─────────────────────────────────────────────────────────────
def train_spoilage_model(lot_df):
    """
    Trains a Random Forest to predict if a lot will spoil.

    Why Random Forest?
    - Handles mixed features (numerical + categorical) well
    - Robust to outliers (lot quantities vary a lot)
    - Feature importance is easy to explain in interviews
    - Doesn't need feature scaling
    """
    print("\n🌲 Training Spoilage Risk Model (Random Forest)...")

    # ── Feature Engineering ──
    le_category = LabelEncoder()
    le_warehouse = LabelEncoder()

    df = lot_df.copy()
    df["category_enc"]  = le_category.fit_transform(df["category"])
    df["warehouse_enc"] = le_warehouse.fit_transform(df["warehouse_id"])

    # Features the model uses to predict spoilage
    FEATURE_COLS = [
        "age_pct",               # How old is the lot? (0=fresh, 1=at expiry)
        "temp_deviation_c",      # How much did temp deviate from ideal?
        "over_order_factor",     # Was this over-ordered?
        "quantity_kg",           # Batch size
        "shelf_life_days",       # How perishable is this category?
        "price_per_kg",          # Higher price items = more care needed
        "category_enc",          # Category (encoded)
        "warehouse_enc",         # Warehouse (encoded)
    ]

    X = df[FEATURE_COLS].values
    y = df["did_spoil"].values

    # ── Train / Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # ── Train Model ──
    # class_weight='balanced' because spoilage is a minority class
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)[:, 1]  # probability of spoiling

    accuracy    = (y_pred == y_test).mean()
    f1          = f1_score(y_test, y_pred)
    report      = classification_report(y_test, y_pred, target_names=["No Spoilage", "Spoilage"])
    cm          = confusion_matrix(y_test, y_pred)

    print(f"   Accuracy : {accuracy:.1%}")
    print(f"   F1 Score : {f1:.3f}")
    print(f"\n   Classification Report:\n{report}")
    print(f"   Confusion Matrix:\n{cm}")

    # ── Feature Importance ──
    feat_imp = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\n   Feature Importances:\n{feat_imp.to_string(index=False)}")

    # Save metrics for dashboard
    metrics = {
        "accuracy":   round(accuracy, 4),
        "f1_score":   round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "feature_importance": feat_imp.to_dict("records"),
        "feature_cols": FEATURE_COLS,
        "report": report,
    }

    return model, le_category, le_warehouse, metrics, feat_imp


# ─────────────────────────────────────────────────────────────
# MODEL 2: DEMAND FORECASTING
# ─────────────────────────────────────────────────────────────
def build_lag_features(df_sku, n_lags=7):
    """
    Builds lag + rolling features for a single SKU's time series.
    This is the feature engineering step for demand forecasting.
    """
    df = df_sku.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Lag features: demand from past N days
    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["actual_demand_kg"].shift(lag)

    # Rolling statistics
    df["rolling_7_mean"]  = df["actual_demand_kg"].shift(1).rolling(7).mean()
    df["rolling_7_std"]   = df["actual_demand_kg"].shift(1).rolling(7).std()
    df["rolling_14_mean"] = df["actual_demand_kg"].shift(1).rolling(14).mean()

    # Calendar features
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["day_of_year"]  = df["date"].dt.dayofyear
    df["is_weekend"]   = (df["date"].dt.dayofweek >= 5).astype(int)

    return df.dropna()


def train_demand_model(demand_df):
    """
    Trains a Gradient Boosting model per SKU for demand forecasting.

    Why Gradient Boosting (GBM)?
    - Captures non-linear effects (festivals, weather, weekends)
    - Works well with the lag + calendar features we created
    - More powerful than ARIMA for complex patterns
    - In production, we'd upgrade to LightGBM or DeepAR
    """
    print("\n📈 Training Demand Forecasting Models (Gradient Boosting per SKU)...")

    sku_models  = {}
    sku_metrics = {}

    FEATURE_COLS = [
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
        "rolling_7_mean", "rolling_7_std", "rolling_14_mean",
        "day_of_week", "month", "day_of_year", "is_weekend",
        "temp_c", "festival_multiplier",
    ]

    skus_in_data = demand_df["sku_id"].unique()

    for sku_id in skus_in_data:
        df_sku    = demand_df[demand_df["sku_id"] == sku_id].copy()
        df_feats  = build_lag_features(df_sku)

        if len(df_feats) < 50:
            continue

        available_features = [f for f in FEATURE_COLS if f in df_feats.columns]
        X = df_feats[available_features].values
        y = df_feats["actual_demand_kg"].values

        # Use last 30 days as test set
        split = max(30, int(len(X) * 0.85))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=RANDOM_SEED
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # demand can't be negative

        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        sku_models[sku_id]  = {"model": model, "features": available_features}
        sku_metrics[sku_id] = {
            "mape": round(mape, 2),
            "test_actual":    y_test.tolist(),
            "test_predicted": y_pred.tolist(),
        }

    avg_mape = np.mean([m["mape"] for m in sku_metrics.values()])
    print(f"   ✅ Trained models for {len(sku_models)} SKUs")
    print(f"   Average MAPE: {avg_mape:.1f}%")
    return sku_models, sku_metrics


# ─────────────────────────────────────────────────────────────
# NEWSVENDOR: Optimal Order Quantity
# ─────────────────────────────────────────────────────────────
def compute_optimal_order(predicted_demand, std_demand,
                          cu=0.25, co=1.00):
    """
    Newsvendor model: finds optimal order quantity.

    cu = cost of under-ordering (stockout cost): lost margin + trust
    co = cost of over-ordering (spoilage cost): 100% loss

    Critical ratio = cu / (cu + co)
    Order at the critical-ratio quantile of the demand distribution.

    Interview explanation:
    'We don't just forecast mean demand — we use the demand distribution
     to find the quantity that minimises TOTAL cost (wastage + stockout).'
    """
    from scipy import stats
    critical_ratio = cu / (cu + co)  # = 0.20 with defaults

    # Demand follows approx. normal distribution
    optimal_qty = stats.norm.ppf(critical_ratio,
                                  loc=predicted_demand,
                                  scale=max(std_demand, 0.01))
    return max(0, round(optimal_qty, 1))


# ─────────────────────────────────────────────────────────────
# GENERATE FORECASTS for the Dashboard
# ─────────────────────────────────────────────────────────────
def generate_forecast(demand_df, sku_id, warehouse_id, horizon=14,
                       sku_models=None):
    """
    Generates a 14-day demand forecast for a given SKU + warehouse.
    Uses the trained GBM model when available, falls back to heuristic.
    Returns a DataFrame with date, forecast, upper/lower bounds.
    """
    df_sku = demand_df[
        (demand_df["sku_id"] == sku_id) &
        (demand_df["warehouse_id"] == warehouse_id)
    ].copy()

    if len(df_sku) < 30:
        return None

    df_sku = df_sku.sort_values("date").reset_index(drop=True)
    df_sku["date"] = pd.to_datetime(df_sku["date"])

    last_date    = df_sku["date"].max()
    last_values  = df_sku["actual_demand_kg"].values[-30:]
    rolling_mean = last_values.mean()
    rolling_std  = last_values.std()

    # ── Try using trained GBM model ──
    use_model = (sku_models is not None and sku_id in sku_models)

    if use_model:
        model_info = sku_models[sku_id]
        model      = model_info["model"]
        feat_names = model_info["features"]

        # Build a working copy with lag features using the last available data
        recent = df_sku.tail(60).copy()  # need enough history for lags
        demand_series = recent["actual_demand_kg"].values.tolist()

    forecasts = []
    for i in range(1, horizon + 1):
        forecast_date = last_date + pd.Timedelta(days=i)

        if use_model:
            # Construct feature vector from current demand_series
            feat_dict = {}
            n = len(demand_series)
            for lag in [1, 2, 3, 7, 14]:
                feat_dict[f"lag_{lag}"] = demand_series[n - lag] if n >= lag else rolling_mean

            last_7  = demand_series[max(0, n-7):]
            last_14 = demand_series[max(0, n-14):]
            feat_dict["rolling_7_mean"]  = float(np.mean(last_7))
            feat_dict["rolling_7_std"]   = float(np.std(last_7)) if len(last_7) > 1 else 1.0
            feat_dict["rolling_14_mean"] = float(np.mean(last_14))
            feat_dict["day_of_week"]     = forecast_date.dayofweek
            feat_dict["month"]           = forecast_date.month
            feat_dict["day_of_year"]     = forecast_date.dayofyear
            feat_dict["is_weekend"]      = int(forecast_date.dayofweek >= 5)
            feat_dict["temp_c"]          = recent["temp_c"].values[-1] if "temp_c" in recent.columns else 28.0
            feat_dict["festival_multiplier"] = 1.0  # default; could be looked up

            X_row = np.array([[feat_dict.get(f, 0) for f in feat_names]])
            base_forecast = max(0, float(model.predict(X_row)[0]))

            # Append predicted value for next iteration's lag features
            demand_series.append(base_forecast)
        else:
            # Fallback heuristic: rolling mean + dampened trend
            recent_14 = df_sku["actual_demand_kg"].values[-14:]
            trend     = (recent_14[-1] - recent_14[0]) / 14 * 0.3
            dow_factors = {0: 0.75, 1: 0.90, 2: 1.00, 3: 1.05,
                           4: 1.10, 5: 1.20, 6: 1.15}
            dow_mult = dow_factors.get(forecast_date.dayofweek, 1.0)
            base_forecast = max(0, (rolling_mean + trend * i) * dow_mult)

        # Uncertainty widens over time
        uncertainty = rolling_std * (1 + i * 0.04)

        forecasts.append({
            "date":            forecast_date.strftime("%Y-%m-%d"),
            "forecast_kg":     round(base_forecast, 1),
            "lower_bound":     round(max(0, base_forecast - 1.5 * uncertainty), 1),
            "upper_bound":     round(base_forecast + 1.5 * uncertainty, 1),
            "optimal_order":   compute_optimal_order(base_forecast, uncertainty),
            "type":            "Forecast",
        })

    return pd.DataFrame(forecasts)


# ─────────────────────────────────────────────────────────────
# MAIN — Train and save all models
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("\n🤖 Hyperpure PISA — Model Trainer")
    print("=" * 45)

    # Load data
    print("\n📂 Loading data...")
    lot_df    = pd.read_csv("data/lot_data.csv")
    demand_df = pd.read_csv("data/demand_data.csv")
    print(f"   Lot data:    {len(lot_df):,} rows")
    print(f"   Demand data: {len(demand_df):,} rows")

    # Train spoilage model
    spoilage_model, le_cat, le_wh, spoilage_metrics, feat_imp = train_spoilage_model(lot_df)

    # Train demand models
    demand_models, demand_metrics = train_demand_model(demand_df)

    # Save everything
    print("\n💾 Saving models...")
    with open("models/spoilage_model.pkl", "wb") as f:
        pickle.dump({
            "model":          spoilage_model,
            "le_category":    le_cat,
            "le_warehouse":   le_wh,
            "metrics":        spoilage_metrics,
            "feature_importance": feat_imp,
        }, f)

    with open("models/demand_models.pkl", "wb") as f:
        pickle.dump({
            "models":  demand_models,
            "metrics": demand_metrics,
        }, f)

    avg_mape = np.mean([m["mape"] for m in demand_metrics.values()])
    print(f"   ✅ Spoilage model saved  (F1: {spoilage_metrics['f1_score']:.3f})")
    print(f"   ✅ Demand models saved   (Avg MAPE: {avg_mape:.1f}%)")
    print("\n🎉 Model training complete!\n")
