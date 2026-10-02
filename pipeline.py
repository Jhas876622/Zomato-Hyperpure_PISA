# =============================================================
# pipeline.py — Nightly ETL + retraining pipeline
#
#   1. Ingest       → raw CSVs (synthetic WMS orders, lots, IoT sensors, vendors)
#   2. Drift check  → reports/drift_report.json
#   3. Train        → only if drift detected, models missing, or --force-train
#   4. Score        → model risk scores on today's active lots
#   5. Load         → rebuild the DuckDB star schema + marts
#
# Run: python pipeline.py [--force-train] [--skip-ingest]
# Scheduled nightly by .github/workflows/nightly-pipeline.yml
# (ponytail: GitHub Actions cron instead of Airflow; move to an Airflow DAG when steps need retries/backfills)
# =============================================================

import argparse
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd

import data_generator
import ml_models
from config import DATA_DIR, MODELS_DIR, REPORTS_DIR
from drift import drift_report
from warehouse_db import build_warehouse


def step(name):
    print(f"\n> {name}")
    return time.time()


def run(force_train=False, skip_ingest=False):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    summary = {"started_at": datetime.now(timezone.utc).isoformat()}

    if not skip_ingest:
        t = step("Ingest")
        data_generator.generate_all()
        summary["ingest_s"] = round(time.time() - t, 1)

    demand_df = pd.read_csv(os.path.join(DATA_DIR, "demand_data.csv"))
    lot_df    = pd.read_csv(os.path.join(DATA_DIR, "lot_data.csv"))

    t = step("Drift check")
    drift = drift_report(demand_df, lot_df)
    with open(os.path.join(REPORTS_DIR, "drift_report.json"), "w") as f:
        json.dump(drift, f, indent=2)
    print(f"   Drifted features: {drift['drifted_features'] or 'none'}")
    summary["drifted_features"] = drift["drifted_features"]

    models_missing = not os.path.exists(os.path.join(MODELS_DIR, "spoilage_model.pkl"))
    train_reason = ("forced" if force_train else "models missing" if models_missing
                    else "drift" if drift["retrain_recommended"] else None)
    if train_reason:
        t = step(f"Train ({train_reason})")
        sp_artifact, dm_artifact = ml_models.train_all(lot_df, demand_df)
        summary["train_s"] = round(time.time() - t, 1)
    else:
        print("\n> Train - skipped (no drift, models present)")
        sp_artifact, dm_artifact = ml_models.load_artifacts()
    summary["trained"] = train_reason

    t = step("Score active lots")
    active_path = os.path.join(DATA_DIR, "active_lots.csv")
    scored = ml_models.score_active_lots(pd.read_csv(active_path), sp_artifact)
    scored.to_csv(active_path, index=False)
    summary["risk_counts"] = scored["risk_level"].value_counts().to_dict()
    print(f"   {summary['risk_counts']}")

    t = step("Load warehouse")
    build_warehouse()

    summary["metrics"] = {
        "spoilage_f1":      sp_artifact["metrics"]["f1_score"],
        "survival_c_index": sp_artifact["survival_metrics"]["c_index"],
        "demand_avg_mape":  round(sum(m["mape"] for m in dm_artifact["metrics"].values())
                                  / len(dm_artifact["metrics"]), 2),
    }
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(REPORTS_DIR, "pipeline_run.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[OK] Pipeline complete")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PISA nightly pipeline")
    ap.add_argument("--force-train", action="store_true", help="retrain even without drift")
    ap.add_argument("--skip-ingest", action="store_true", help="reuse existing CSVs")
    args = ap.parse_args()
    run(force_train=args.force_train, skip_ingest=args.skip_ingest)
