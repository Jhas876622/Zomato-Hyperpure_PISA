# =============================================================
# api.py — PISA REST API (FastAPI)
#
# Run:  uvicorn api:app --reload        Docs: http://localhost:8000/docs
# Auth: every endpoint except /health requires header  X-API-Key: <PISA_API_KEY>.
#       Fails closed: with PISA_ENV=production (set in the Dockerfile) and no
#       PISA_API_KEY, protected endpoints return 503 instead of running open.
#       Locally (PISA_ENV unset) a missing key leaves the API open for development.
# Limits: RATE_LIMIT_PER_MIN requests per client IP; standard security headers.
# HTTPS: terminate TLS in front of this service (reverse proxy / platform).
# =============================================================

import json
import os
import secrets
import time
from collections import defaultdict, deque
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

import ml_models
from config import DATA_DIR, REPORTS_DIR, SKUS, WAREHOUSES

app = FastAPI(title="PISA API", version="1.0",
              description="Predictive Inventory & Spoilage Alerts — Zomato Hyperpure B2B")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
RATE_LIMIT_PER_MIN = int(os.getenv("PISA_RATE_LIMIT_PER_MIN", "60"))

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",  # honoured only over HTTPS
}

# ponytail: in-memory, per-process limiter; use a shared store (e.g. Redis) if you run >1 worker
_hits = defaultdict(deque)


@app.middleware("http")
async def rate_limit_and_headers(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now, window = time.monotonic(), _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        response = JSONResponse({"detail": "Too many requests, slow down."}, status_code=429,
                                headers={"Retry-After": "60"})
    else:
        window.append(now)
        response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    return response


def require_api_key(key: str | None = Security(api_key_header)):
    expected = os.getenv("PISA_API_KEY")
    if not expected:
        if os.getenv("PISA_ENV") == "production":
            raise HTTPException(status_code=503, detail="API key not configured on the server")
        return  # local development
    if not (key and secrets.compare_digest(key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@lru_cache(maxsize=1)
def artifacts():
    return ml_models.load_artifacts()


@lru_cache(maxsize=1)
def demand_df():
    return pd.read_csv(os.path.join(DATA_DIR, "demand_data.csv"))


SKU_IDS = {s["sku_id"] for s in SKUS}
WH_IDS  = {w["id"] for w in WAREHOUSES}
SKU_BY_ID = {s["sku_id"]: s for s in SKUS}


# ── Schemas ───────────────────────────────────────────────────
class LotIn(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields are rejected, not silently ignored
    sku_id: str = Field(max_length=16, examples=["SKU001"])
    warehouse_id: str = Field(max_length=16, examples=["WH_DEL_01"])
    days_old: int = Field(ge=0, le=60)
    quantity_kg: float = Field(gt=0, le=100_000)
    temp_deviation_c: float = Field(ge=0, le=30)
    over_order_factor: float = Field(default=1.2, ge=0.5, le=5)
    vendor_reject_rate: float = Field(default=0.05, ge=0, le=1)


class LotRisk(BaseModel):
    risk_score: int
    risk_level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    spoil_prob_48h_pct: float
    recommended_action: str


class NewsvendorIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean_demand_kg: float = Field(ge=0, le=1_000_000)
    std_demand_kg: float = Field(ge=0, le=1_000_000)
    cost_understock: float = Field(default=0.25, gt=0, le=100)
    cost_overstock: float = Field(default=1.00, gt=0, le=100)


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        sp, dm = artifacts()
        return {"status": "ok", "demand_models": len(dm["models"]),
                "spoilage_f1": sp["metrics"]["f1_score"]}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Models not trained yet — run python pipeline.py")


ALERT_FIELDS = ["lot_id", "sku_id", "sku_name", "category", "warehouse_id", "warehouse_name",
                "days_remaining", "quantity_kg", "lot_value_inr", "risk_score", "risk_level",
                "spoil_prob_48h", "recommended_action"]


@app.get("/forecast/{sku_id}/{warehouse_id}", dependencies=[Depends(require_api_key)])
def forecast(sku_id: str, warehouse_id: str, horizon: int = Query(14, ge=1, le=30)):
    if sku_id not in SKU_IDS or warehouse_id not in WH_IDS:
        raise HTTPException(status_code=404, detail="Unknown sku_id or warehouse_id")
    _, dm = artifacts()
    fc = ml_models.generate_forecast(demand_df(), sku_id, warehouse_id, horizon, sku_models=dm["models"])
    if fc is None:
        raise HTTPException(status_code=404, detail="Not enough history for this SKU/warehouse")
    fc = fc.drop(columns=["temp_source", "type"], errors="ignore")
    return {"sku_id": sku_id, "warehouse_id": warehouse_id, "forecast": fc.to_dict("records")}


@app.post("/spoilage/score", response_model=LotRisk, dependencies=[Depends(require_api_key)])
def score_lot(lot: LotIn):
    sku = SKU_BY_ID.get(lot.sku_id)
    if sku is None or lot.warehouse_id not in WH_IDS:
        raise HTTPException(status_code=404, detail="Unknown sku_id or warehouse_id")
    row = pd.DataFrame([{
        **lot.model_dump(),
        "category": sku["category"], "shelf_life_days": sku["shelf_life_days"],
        "price_per_kg": sku["price_per_kg"],
        "age_pct": min(1.0, lot.days_old / sku["shelf_life_days"]),
    }])
    sp, _ = artifacts()
    r = ml_models.score_active_lots(row, sp).iloc[0]
    return LotRisk(risk_score=int(r.risk_score), risk_level=r.risk_level,
                   spoil_prob_48h_pct=float(r.spoil_prob_48h),
                   recommended_action=r.recommended_action)


@app.get("/alerts", dependencies=[Depends(require_api_key)])
def alerts(min_level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = "HIGH"):
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    sp, _ = artifacts()
    lots = ml_models.score_active_lots(pd.read_csv(os.path.join(DATA_DIR, "active_lots.csv")), sp)
    keep = lots[lots["risk_level"].isin(order[: order.index(min_level) + 1])]
    fields = [c for c in ALERT_FIELDS if c in keep.columns]  # only what a client needs
    return {"count": len(keep), "lots": keep[fields].to_dict("records")}


@app.post("/newsvendor", dependencies=[Depends(require_api_key)])
def newsvendor(req: NewsvendorIn):
    qty = ml_models.compute_optimal_order(req.mean_demand_kg, req.std_demand_kg,
                                          cu=req.cost_understock, co=req.cost_overstock)
    return {"optimal_order_kg": qty,
            "critical_ratio": round(req.cost_understock / (req.cost_understock + req.cost_overstock), 3)}


@app.get("/monitoring/drift", dependencies=[Depends(require_api_key)])
def drift():
    path = os.path.join(REPORTS_DIR, "drift_report.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No drift report yet — run python pipeline.py")
    with open(path) as f:
        return json.load(f)
