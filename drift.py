# =============================================================
# drift.py — Data drift monitoring
#
# Compares the most recent window of data against the reference
# (training) window, feature by feature:
#   - Kolmogorov–Smirnov test  (is the distribution different?)
#   - Population Stability Index (how much did it shift?)
# A feature drifts when PSI > 0.2 (industry rule of thumb) AND KS p < 0.05.
# =============================================================

import numpy as np
import pandas as pd
from scipy import stats

PSI_THRESHOLD = 0.2
P_VALUE       = 0.05

# ponytail: KS + PSI with scipy instead of Evidently; switch to Evidently if you need its HTML reports
DEMAND_FEATURES = ["actual_demand_kg", "temp_c", "festival_multiplier"]
LOT_FEATURES    = ["temp_deviation_c", "over_order_factor", "quantity_kg", "vendor_reject_rate"]


def psi(reference, current, bins=10):
    """Population Stability Index over reference-quantile bins."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: nothing to bin
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-4, None), np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compare(reference_df, current_df, features):
    rows = []
    for f in features:
        ref, cur = reference_df[f].dropna().to_numpy(), current_df[f].dropna().to_numpy()
        ks = stats.ks_2samp(ref, cur)
        p  = psi(ref, cur)
        rows.append({"feature": f, "psi": round(p, 3), "ks_stat": round(float(ks.statistic), 3),
                     "p_value": round(float(ks.pvalue), 4),
                     "drifted": bool(p > PSI_THRESHOLD and ks.pvalue < P_VALUE)})
    return rows


def drift_report(demand_df, lot_df, window_days=30):
    """Last `window_days` vs everything before it, for orders and lots."""
    d_date = pd.to_datetime(demand_df["date"])
    d_cut  = d_date.max() - pd.Timedelta(days=window_days)
    l_date = pd.to_datetime(lot_df["procurement_date"])
    l_cut  = l_date.max() - pd.Timedelta(days=window_days)

    features = (compare(demand_df[d_date <= d_cut], demand_df[d_date > d_cut], DEMAND_FEATURES)
                + compare(lot_df[l_date <= l_cut], lot_df[l_date > l_cut], LOT_FEATURES))
    drifted = [f["feature"] for f in features if f["drifted"]]
    return {"window_days": window_days, "features": features,
            "drifted_features": drifted, "retrain_recommended": bool(drifted)}


if __name__ == "__main__":
    import json
    report = drift_report(pd.read_csv("data/demand_data.csv"), pd.read_csv("data/lot_data.csv"))
    print(json.dumps(report, indent=2))
