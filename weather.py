# =============================================================
# weather.py — OpenWeatherMap forecast client
#
# Set OPENWEATHER_API_KEY to use live 5-day forecasts. Without a key
# (or on any API failure) we fall back to the same seasonal temperature
# curve the synthetic data was generated with, so forecasts never break.
# =============================================================

import os
from datetime import date
from functools import lru_cache

import numpy as np
import requests

OWM_URL = "https://api.openweathermap.org/data/2.5/forecast"


def seasonal_temp(d):
    """Climatology fallback — matches data_generator's temperature curve."""
    return round(28 + 7 * np.sin(2 * np.pi * (d.timetuple().tm_yday - 30) / 365), 1)


@lru_cache(maxsize=32)
def _fetch_daily_means(city, today_iso):
    # today_iso is only part of the cache key, so results refresh once a day
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return {}
    try:
        resp = requests.get(OWM_URL, params={"q": f"{city},IN", "units": "metric", "appid": key}, timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️  Weather API unavailable for {city}: {e}")
        return {}

    # 3-hourly readings → daily mean temperature
    by_day = {}
    for item in resp.json().get("list", []):
        by_day.setdefault(item["dt_txt"][:10], []).append(item["main"]["temp"])
    return {d: round(float(np.mean(t)), 1) for d, t in by_day.items()}


def get_temp_forecast(city, dates):
    """Returns {date: (temp_c, source)} for each date — 'api' or 'seasonal'."""
    live = _fetch_daily_means(city, date.today().isoformat()) if city else {}
    out = {}
    for d in dates:
        iso = d.strftime("%Y-%m-%d")
        out[d] = (live[iso], "api") if iso in live else (seasonal_temp(d), "seasonal")
    return out
