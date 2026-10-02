# =============================================================
# ml_models.py — ML Model Training for PISA
#
# Engine 1: Demand Forecasting (XGBoost, one model per SKU)
#   → Predicts: How much of SKU X will be ordered tomorrow?
#   → Uses: lag features, day-of-week, festivals, weather
#
# Engine 2: Spoilage Risk
#   → Random Forest: Will this lot spoil? (risk score 0-100)
#   → Cox Proportional Hazards: WHEN will it spoil? (P(spoil in next 48h))
#
# Engine 3: Newsvendor optimal order quantity
#
# Training runs are tracked in MLflow when it is installed (requirements-dev.txt);
# otherwise tracking is skipped.
# Run standalone: python ml_models.py
# =============================================================

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, mean_absolute_percentage_error, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from config import (RANDOM_SEED, RISK_LEVELS, MODELS_DIR,
                    COST_OF_OVERSTOCKING_PCT, COST_OF_UNDERSTOCKING_PCT)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence MLflow's import-time log line
try:
    import mlflow  # optional: the hosted Streamlit demo trains without it
except ImportError:
    mlflow = None

np.random.seed(RANDOM_SEED)

SPOILAGE_FEATURES = [
    "age_pct",               # How old is the lot? (0=fresh, 1=at expiry)
    "temp_deviation_c",      # Cold-chain deviation from ideal temp
    "over_order_factor",     # Was this over-ordered?
    "quantity_kg",           # Batch size
    "shelf_life_days",       # How perishable is this category?
    "price_per_kg",          # Higher price items = more care needed
    "vendor_reject_rate",    # Inbound quality of the supplier
    "category_enc",          # Category (encoded)
    "warehouse_enc",         # Warehouse (encoded)
]
# Cox covariates: lot attributes known at receipt (age is the time axis itself)
COX_FEATURES = ["temp_deviation_c", "over_order_factor", "vendor_reject_rate", "shelf_life_days"]

DEMAND_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
    "rolling_7_mean", "rolling_7_std", "rolling_14_mean",
    "day_of_week", "month", "day_of_year", "is_weekend",
    "temp_c", "festival_multiplier",
]
TEST_DAYS = 30


# ─────────────────────────────────────────────────────────────
# ENGINE 2a: SPOILAGE RISK CLASSIFIER (Random Forest)
# ─────────────────────────────────────────────────────────────
def train_spoilage_model(lot_df):
    """
    Trains a Random Forest to predict if a lot will spoil.

    Why Random Forest?
    - Handles mixed features (numerical + categorical) well
    - Robust to outliers (lot quantities vary a lot)
    - Feature importance is easy to explain
    - Doesn't need feature scaling
    """
    print("\n🌲 Training Spoilage Risk Model (Random Forest)...")

    le_category  = LabelEncoder().fit(lot_df["category"])
    le_warehouse = LabelEncoder().fit(lot_df["warehouse_id"])

    df = lot_df.copy()
    df["category_enc"]  = le_category.transform(df["category"])
    df["warehouse_enc"] = le_warehouse.transform(df["warehouse_id"])

    X = df[SPOILAGE_FEATURES].values
    y = df["did_spoil"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # class_weight='balanced' because spoilage is a minority class
    model = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = (y_pred == y_test).mean()
    f1       = f1_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_proba)
    report   = classification_report(y_test, y_pred, target_names=["No Spoilage", "Spoilage"])
    cm       = confusion_matrix(y_test, y_pred)

    print(f"   Accuracy : {accuracy:.1%}   F1: {f1:.3f}   ROC-AUC: {auc:.3f}")

    feat_imp = pd.DataFrame({
        "feature":    SPOILAGE_FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    metrics = {
        "accuracy":   round(accuracy, 4),
        "f1_score":   round(f1, 4),
        "roc_auc":    round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "feature_importance": feat_imp.to_dict("records"),
        "feature_cols": SPOILAGE_FEATURES,
        "report": report,
    }
    return model, le_category, le_warehouse, metrics, feat_imp


# ─────────────────────────────────────────────────────────────
# ENGINE 2b: TIME-TO-SPOILAGE (Cox Proportional Hazards)
# ─────────────────────────────────────────────────────────────
def train_survival_model(lot_df):
    """
    Cox PH survival model: duration = days until the lot spoiled,
    censored at end of shelf life for lots that were dispatched fine.

    Why survival analysis on top of the classifier?
    The RF says IF a lot is risky; Cox says WHEN — so the warehouse
    can act on lots likely to spoil in the next 48h first.
    """
    print("\n⏳ Training Time-to-Spoilage Model (Cox PH)...")
    df = lot_df[COX_FEATURES + ["days_to_event", "did_spoil"]].copy()
    train, test = train_test_split(df, test_size=0.20, random_state=RANDOM_SEED)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(train, duration_col="days_to_event", event_col="did_spoil")
    c_index = cph.score(test, scoring_method="concordance_index")
    print(f"   Concordance index (test): {c_index:.3f}")

    metrics = {
        "c_index": round(float(c_index), 4),
        "hazard_ratios": cph.hazard_ratios_.round(3).to_dict(),
    }
    return cph, metrics


def spoil_prob_within(cph, lots, horizon_days=2):
    """P(spoil within next `horizon_days` | survived until today), per lot."""
    t_now  = lots["days_old"].clip(lower=0).to_numpy(dtype=float)
    max_t  = float(lots["days_old"].max()) + horizon_days + 1
    times  = np.arange(0, max_t + 1)
    surv   = cph.predict_survival_function(lots[COX_FEATURES], times=times).to_numpy()  # (times, lots)

    idx = np.arange(len(lots))
    s_now   = surv[t_now.astype(int), idx]
    s_later = surv[(t_now + horizon_days).astype(int), idx]
    return np.clip(1 - s_later / np.maximum(s_now, 1e-6), 0, 1)


# ─────────────────────────────────────────────────────────────
# ENGINE 1: DEMAND FORECASTING (XGBoost)
# ─────────────────────────────────────────────────────────────
def build_lag_features(df_series, n_lags=7):
    """
    Builds lag + rolling features for ONE (SKU, warehouse) time series.
    Callers must not pass several warehouses at once, or lag_1 would be
    another warehouse's demand.
    """
    df = df_series.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["actual_demand_kg"].shift(lag)

    df["rolling_7_mean"]  = df["actual_demand_kg"].shift(1).rolling(7).mean()
    df["rolling_7_std"]   = df["actual_demand_kg"].shift(1).rolling(7).std()
    df["rolling_14_mean"] = df["actual_demand_kg"].shift(1).rolling(14).mean()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"]  = (df["date"].dt.dayofweek >= 5).astype(int)

    return df.dropna()


def train_demand_model(demand_df):
    """
    Trains one XGBoost regressor per SKU (pooled over warehouses).

    Why XGBoost?
    - Captures non-linear effects (festivals, weather, weekends)
    - Regularised boosting → less overfitting than plain GBM
    - Fast, and the industry default for tabular forecasting

    Evaluation is a time-based split: the last TEST_DAYS days are held out,
    and every SKU is compared to a seasonal-naive baseline (same day last week).
    """
    print("\n📈 Training Demand Forecasting Models (XGBoost per SKU)...")

    cutoff = pd.to_datetime(demand_df["date"]).max() - pd.Timedelta(days=TEST_DAYS)
    sku_models, sku_metrics = {}, {}

    for sku_id, df_sku in demand_df.groupby("sku_id"):
        # Lags are built per warehouse, then pooled
        feats = pd.concat([build_lag_features(g) for _, g in df_sku.groupby("warehouse_id")])
        if len(feats) < 50:
            continue

        train = feats[feats["date"] <= cutoff]
        test  = feats[feats["date"] > cutoff].sort_values(["warehouse_id", "date"])

        model = XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=1,
        )
        model.fit(train[DEMAND_FEATURES], train["actual_demand_kg"])

        y_test = test["actual_demand_kg"].to_numpy()
        y_pred = np.maximum(model.predict(test[DEMAND_FEATURES]), 0)  # demand can't be negative

        mape          = mean_absolute_percentage_error(y_test, y_pred) * 100
        baseline_mape = mean_absolute_percentage_error(y_test, test["lag_7"]) * 100

        # Plot series: first warehouse only, so the line is one continuous series
        first_wh = test["warehouse_id"] == test["warehouse_id"].iloc[0]
        sku_models[sku_id]  = {"model": model, "features": DEMAND_FEATURES}
        sku_metrics[sku_id] = {
            "mape":          round(mape, 2),
            "baseline_mape": round(baseline_mape, 2),
            "residual_std":  round(float(np.std(y_test - y_pred)), 2),
            "test_actual":    y_test[first_wh.to_numpy()].tolist(),
            "test_predicted": y_pred[first_wh.to_numpy()].tolist(),
        }

    avg_mape = np.mean([m["mape"] for m in sku_metrics.values()])
    avg_base = np.mean([m["baseline_mape"] for m in sku_metrics.values()])
    print(f"   ✅ Trained models for {len(sku_models)} SKUs")
    print(f"   Average MAPE: {avg_mape:.1f}%  (seasonal-naive baseline: {avg_base:.1f}%)")
    return sku_models, sku_metrics


# ─────────────────────────────────────────────────────────────
# ENGINE 3: NEWSVENDOR — Optimal Order Quantity
# ─────────────────────────────────────────────────────────────
def compute_optimal_order(predicted_demand, std_demand,
                          cu=COST_OF_UNDERSTOCKING_PCT, co=COST_OF_OVERSTOCKING_PCT):
    """
    Newsvendor model: finds optimal order quantity.

    cu = cost of under-ordering (stockout cost): lost margin + trust
    co = cost of over-ordering (spoilage cost): 100% loss

    Critical ratio = cu / (cu + co); order at that quantile of the
    demand distribution (approx. normal).
    """
    critical_ratio = cu / (cu + co)  # = 0.20 with defaults
    optimal_qty = stats.norm.ppf(critical_ratio, loc=predicted_demand, scale=max(std_demand, 0.01))
    return max(0, round(float(optimal_qty), 1))


# ─────────────────────────────────────────────────────────────
# SCORE LIVE LOTS with the trained spoilage models
# ─────────────────────────────────────────────────────────────
ACTIONS = {
    "CRITICAL": "Redistribute or discount immediately (< 12h)",
    "HIGH":     "Offer 25% discount to bulk buyers today",
    "MEDIUM":   "Prioritise dispatch in next shipment",
    "LOW":      "Monitor; on track for normal dispatch",
}


def risk_level(score):
    return next(level for threshold, level in RISK_LEVELS if score >= threshold)


def score_active_lots(active_lots, sp_artifact):
    """Adds risk_score, risk_level, spoil_prob_48h and recommended_action."""
    df = active_lots.copy()
    # Unseen categories/warehouses fall back to code 0 rather than crashing
    for col, enc_col, le in [("category", "category_enc", sp_artifact["le_category"]),
                             ("warehouse_id", "warehouse_enc", sp_artifact["le_warehouse"])]:
        mapping = {c: i for i, c in enumerate(le.classes_)}
        df[enc_col] = df[col].map(mapping).fillna(0).astype(int)

    proba = sp_artifact["model"].predict_proba(df[SPOILAGE_FEATURES].to_numpy())[:, 1]
    df["risk_score"] = np.round(proba * 100).astype(int)
    df["risk_level"] = df["risk_score"].map(risk_level)
    df["recommended_action"] = df["risk_level"].map(ACTIONS)
    if sp_artifact.get("survival_model") is not None:
        df["spoil_prob_48h"] = np.round(spoil_prob_within(sp_artifact["survival_model"], df) * 100, 1)
    return df.drop(columns=["category_enc", "warehouse_enc"]).sort_values(
        "risk_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# GENERATE FORECASTS for the Dashboard / API
# ─────────────────────────────────────────────────────────────
def generate_forecast(demand_df, sku_id, warehouse_id, horizon=14, sku_models=None):
    """
    Recursive multi-step forecast for one SKU + warehouse.
    Uses the trained XGBoost model when available, falls back to a heuristic.
    Temperatures come from the weather API (or seasonal climatology).
    Returns a DataFrame with date, forecast, bounds and newsvendor order.
    """
    from weather import get_temp_forecast

    df_sku = demand_df[
        (demand_df["sku_id"] == sku_id) & (demand_df["warehouse_id"] == warehouse_id)
    ].copy()
    if len(df_sku) < 30:
        return None

    df_sku = df_sku.sort_values("date").reset_index(drop=True)
    df_sku["date"] = pd.to_datetime(df_sku["date"])

    last_date    = df_sku["date"].max()
    last_values  = df_sku["actual_demand_kg"].values[-30:]
    rolling_mean = last_values.mean()
    rolling_std  = last_values.std()

    dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    city  = df_sku["city"].iloc[-1] if "city" in df_sku.columns else None
    temps = get_temp_forecast(city, dates)

    use_model = sku_models is not None and sku_id in sku_models
    if use_model:
        model      = sku_models[sku_id]["model"]
        feat_names = sku_models[sku_id]["features"]
        demand_series = df_sku["actual_demand_kg"].values[-60:].tolist()

    forecasts = []
    for i, forecast_date in enumerate(dates, start=1):
        temp_c, temp_source = temps[forecast_date]

        if use_model:
            n = len(demand_series)
            feat = {f"lag_{lag}": demand_series[n - lag] for lag in [1, 2, 3, 7, 14]}
            feat["rolling_7_mean"]  = float(np.mean(demand_series[-7:]))
            feat["rolling_7_std"]   = float(np.std(demand_series[-7:], ddof=1))
            feat["rolling_14_mean"] = float(np.mean(demand_series[-14:]))
            feat["day_of_week"] = forecast_date.dayofweek
            feat["month"]       = forecast_date.month
            feat["day_of_year"] = forecast_date.dayofyear
            feat["is_weekend"]  = int(forecast_date.dayofweek >= 5)
            feat["temp_c"]      = temp_c
            # ponytail: festival calendar only covers the training year; extend config.FESTIVALS for live use
            feat["festival_multiplier"] = 1.0

            X_row = pd.DataFrame([[feat[f] for f in feat_names]], columns=feat_names)
            base_forecast = max(0.0, float(model.predict(X_row)[0]))
            demand_series.append(base_forecast)  # feeds the next step's lags
        else:
            recent_14 = df_sku["actual_demand_kg"].values[-14:]
            trend     = (recent_14[-1] - recent_14[0]) / 14 * 0.3
            dow_mult  = {0: 0.75, 1: 0.90, 2: 1.00, 3: 1.05, 4: 1.10, 5: 1.20, 6: 1.15}[forecast_date.dayofweek]
            base_forecast = max(0.0, (rolling_mean + trend * i) * dow_mult)

        uncertainty = rolling_std * (1 + i * 0.04)  # widens with horizon

        forecasts.append({
            "date":          forecast_date.strftime("%Y-%m-%d"),
            "forecast_kg":   round(base_forecast, 1),
            "lower_bound":   round(max(0, base_forecast - 1.5 * uncertainty), 1),
            "upper_bound":   round(base_forecast + 1.5 * uncertainty, 1),
            "optimal_order": compute_optimal_order(base_forecast, uncertainty),
            "temp_c":        temp_c,
            "temp_source":   temp_source,
            "type":          "Forecast",
        })

    return pd.DataFrame(forecasts)


# ─────────────────────────────────────────────────────────────
# TRAIN EVERYTHING + persist + track in MLflow
# ─────────────────────────────────────────────────────────────
def _log_to_mlflow(sp_artifact, dm_metrics, models_dir):
    """Logs params/metrics/artifacts and registers the spoilage RF as a new model version.
    Tracking URI comes from MLFLOW_TRACKING_URI (default: local sqlite:///mlflow.db)."""
    if mlflow is None:
        print("   (mlflow not installed — skipping experiment tracking)")
        return None
    try:
        from mlflow import sklearn as mlflow_sklearn
        # Explicit relative default: MLflow's own default URI breaks on paths with spaces
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        if mlflow.get_experiment_by_name("pisa") is None:
            mlflow.create_experiment("pisa", artifact_location=os.path.abspath(os.getenv("MLFLOW_ARTIFACT_DIR", "mlartifacts")))
        mlflow.set_experiment("pisa")
        with mlflow.start_run(run_name="nightly-train") as run:
            mlflow.log_params({"rf_n_estimators": 150, "rf_max_depth": 8,
                               "xgb_n_estimators": 150, "xgb_max_depth": 4,
                               "xgb_learning_rate": 0.1, "test_days": TEST_DAYS})
            mlflow.log_metrics({
                "spoilage_f1":          sp_artifact["metrics"]["f1_score"],
                "spoilage_roc_auc":     sp_artifact["metrics"]["roc_auc"],
                "survival_c_index":     sp_artifact["survival_metrics"]["c_index"],
                "demand_avg_mape":      float(np.mean([m["mape"] for m in dm_metrics.values()])),
                "demand_baseline_mape": float(np.mean([m["baseline_mape"] for m in dm_metrics.values()])),
            })
            for f in ("spoilage_model.pkl", "demand_models.pkl"):
                mlflow.log_artifact(os.path.join(models_dir, f))
            mlflow_sklearn.log_model(sp_artifact["model"], name="spoilage_rf",
                                     registered_model_name="pisa-spoilage-rf",
                                     # we trained this RF ourselves, so its tree storage is trusted
                                     skops_trusted_types=["sklearn.tree._tree.Tree"])
            return run.info.run_id
    except Exception as e:  # tracking is best-effort; never lose a trained model over it
        print(f"   ⚠️  MLflow logging failed: {e}")
        return None


def train_all(lot_df, demand_df, models_dir=MODELS_DIR):
    """Trains all engines, saves pickles, logs to MLflow. Returns (sp_artifact, dm_artifact)."""
    os.makedirs(models_dir, exist_ok=True)

    sp_model, le_cat, le_wh, sp_metrics, feat_imp = train_spoilage_model(lot_df)
    cph, sv_metrics = train_survival_model(lot_df)
    dm_models, dm_metrics = train_demand_model(demand_df)

    sp_artifact = {"model": sp_model, "le_category": le_cat, "le_warehouse": le_wh,
                   "metrics": sp_metrics, "feature_importance": feat_imp,
                   "survival_model": cph, "survival_metrics": sv_metrics}
    dm_artifact = {"models": dm_models, "metrics": dm_metrics}

    with open(os.path.join(models_dir, "spoilage_model.pkl"), "wb") as f:
        pickle.dump(sp_artifact, f)
    with open(os.path.join(models_dir, "demand_models.pkl"), "wb") as f:
        pickle.dump(dm_artifact, f)

    run_id = _log_to_mlflow(sp_artifact, dm_metrics, models_dir)
    if run_id:
        print(f"   📒 MLflow run: {run_id}")
    return sp_artifact, dm_artifact


def load_artifacts(models_dir=MODELS_DIR):
    # Pickles are only ever written by train_all() in this repo; never point this at untrusted files.
    with open(os.path.join(models_dir, "spoilage_model.pkl"), "rb") as f:
        sp = pickle.load(f)
    with open(os.path.join(models_dir, "demand_models.pkl"), "rb") as f:
        dm = pickle.load(f)
    return sp, dm


if __name__ == "__main__":
    print("\n🤖 Hyperpure PISA — Model Trainer")
    print("=" * 45)
    lot_df    = pd.read_csv("data/lot_data.csv")
    demand_df = pd.read_csv("data/demand_data.csv")
    train_all(lot_df, demand_df)
    print("\n🎉 Model training complete!\n")
