# =============================================================
# pipeline.py — Nightly ETL + retraining pipeline
#
#   1. Ingest       → raw CSVs (synthetic WMS orders, lots, IoT sensors, vendors)
#   2. Drift check  → reports/drift_report.json
#   3. Train        → only if drift detected, models missing, or --force-train.
#                     The new model is a challenger: it replaces the champion only if
#                     it is not worse (config.PROMOTION_TOLERANCE).
#   4. Score        → model risk scores on today's active lots
#   5. Load         → rebuild the DuckDB star schema + marts
#   6. Check        → alert if a metric is past config.QUALITY_FLOORS
# Any failure, quality breach or rejected challenger is sent to ALERT_WEBHOOK_URL
# (alerts.py); every run writes reports/pipeline_run.json with status ok/failed.
#
# Run: python pipeline.py [--force-train] [--skip-ingest]
# Scheduled nightly by .github/workflows/nightly-pipeline.yml
# (ponytail: GitHub Actions cron instead of Airflow; move to an Airflow DAG when steps need retries/backfills)
# =============================================================

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

import data_generator
import ml_models
from alerts import send_alert
from config import DATA_DIR, MODELS_DIR, PROMOTION_TOLERANCE, QUALITY_FLOORS, REPORTS_DIR
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
        candidate_dir = os.path.join(MODELS_DIR, "candidate")
        sp_c, dm_c = ml_models.train_all(lot_df, demand_df, models_dir=candidate_dir)
        summary["train_s"] = round(time.time() - t, 1)

        champion = ml_models.read_manifest(MODELS_DIR)
        worse = ml_models.regressions(ml_models.summary_metrics(sp_c, dm_c), champion["metrics"],
                                      PROMOTION_TOLERANCE) if champion else []
        summary["promoted"] = not worse
        if worse:
            print(f"   Challenger rejected: {worse}")
            send_alert("Retrained model was worse than the current one and was NOT deployed",
                       worse + ["The previous model stays in service."], level="warning")
        else:
            ml_models.promote(candidate_dir, MODELS_DIR)
            print("   Challenger promoted to champion")
    else:
        print("\n> Train - skipped (no drift, models present)")
    summary["trained"] = train_reason
    sp_artifact, dm_artifact = ml_models.load_artifacts()  # always serve the champion

    t = step("Score active lots")
    active_path = os.path.join(DATA_DIR, "active_lots.csv")
    scored = ml_models.score_active_lots(pd.read_csv(active_path), sp_artifact)
    scored.to_csv(active_path, index=False)
    summary["risk_counts"] = scored["risk_level"].value_counts().to_dict()
    print(f"   {summary['risk_counts']}")

    t = step("Load warehouse")
    build_warehouse()

    summary["metrics"] = ml_models.summary_metrics(sp_artifact, dm_artifact)
    breaches = quality_breaches(summary["metrics"])
    summary["quality_breaches"] = breaches
    if breaches:
        send_alert("Model quality below target", breaches, level="warning")

    summary["status"] = "ok"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_report(summary)
    print("\n[OK] Pipeline complete")
    return summary


def quality_breaches(m):
    f = QUALITY_FLOORS
    out = []
    if m["demand_avg_mape"] > f["demand_avg_mape_max"]:
        out.append(f"Demand MAPE {m['demand_avg_mape']}% is above {f['demand_avg_mape_max']}%")
    if m["spoilage_f1"] < f["spoilage_f1_min"]:
        out.append(f"Spoilage F1 {m['spoilage_f1']:.3f} is below {f['spoilage_f1_min']}")
    if m["survival_c_index"] < f["survival_c_index_min"]:
        out.append(f"Cox C-index {m['survival_c_index']:.3f} is below {f['survival_c_index_min']}")
    return out


def write_report(summary):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "pipeline_run.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main(force_train=False, skip_ingest=False):
    """Runs the pipeline; on any failure records it, alerts, and exits non-zero."""
    started = datetime.now(timezone.utc).isoformat()
    try:
        return run(force_train=force_train, skip_ingest=skip_ingest)
    except Exception as e:
        write_report({"status": "failed", "started_at": started,
                      "finished_at": datetime.now(timezone.utc).isoformat(),
                      "error": f"{type(e).__name__}: {e}"})
        send_alert("Nightly pipeline FAILED", [f"{type(e).__name__}: {e}",
                                               traceback.format_exc().strip().splitlines()[-3]])
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PISA nightly pipeline")
    ap.add_argument("--force-train", action="store_true", help="retrain even without drift")
    ap.add_argument("--skip-ingest", action="store_true", help="reuse existing CSVs")
    args = ap.parse_args()
    try:
        main(force_train=args.force_train, skip_ingest=args.skip_ingest)
    except Exception:
        sys.exit(1)  # main() already printed, recorded and alerted
