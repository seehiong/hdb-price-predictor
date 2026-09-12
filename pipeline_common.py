#!/usr/bin/env python3
"""
pipeline_common.py
Shared helpers for the HDB and private-condo training pipelines.

Two responsibilities:

1. Published metrics. The web app reads its headline accuracy figures from
   models/metrics.json rather than having them written into index.html by hand.
   Each trainer writes only its own section, so the pipelines can be re-run
   independently without clobbering each other's numbers.

2. Market price indices. Gradient-boosted trees cannot extrapolate: a sale_year
   past the training range lands in the same terminal leaf for every future
   year. The fix is to separate *what a property is* (which the model predicts)
   from *what the market is doing* (which an index tracks). We use the official
   published indices rather than a trend fitted to our own transactions:

     - HDB   Resale Price Index (1Q2009=100), quarterly, HDB via data.gov.sg
     - URA   Property Price Index of Non-landed Properties by Locality,
             quarterly, split by CCR / RCR / OCR

   The URA split matters. Over 2021-Q1 to 2026-Q2 the segments diverged sharply
   (CCR +20%, RCR +33%, OCR +46%), so a single blended growth rate overstates
   prime-district forecasts badly - roughly 24% too high six years out.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METRICS_JSON = os.path.join(MODELS_DIR, "metrics.json")

DATASTORE_URL = "https://data.gov.sg/api/action/datastore_search"

# Fallback annual appreciation when a trend cannot be fitted, and a sanity cap
# so a noisy sample can never produce a runaway forecast.
DEFAULT_ANNUAL_GROWTH = 0.025
MAX_ANNUAL_GROWTH = 0.08

# data.gov.sg resource ids for the two official indices.
HDB_RPI_RESOURCE = "d_14f63e595975691e7c24a27ae4c07c79"
URA_PPI_RESOURCE = "d_f65e490a8ad430f60a9a3d9df2bff2a0"

# URA publishes full segment names; the models use the three-letter codes.
SEGMENT_CODES = {
    "Core Central Region": "CCR",
    "Rest of Central Region": "RCR",
    "Outside Central Region": "OCR",
}


def write_metrics(section: str, payload: dict, path: str = METRICS_JSON) -> dict:
    """
    Merges `payload` into `metrics.json` under `section` ("hdb" or "condo").

    Reads the existing file first so retraining one model leaves the other
    model's published figures untouched.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    metrics = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, OSError):
            # A corrupt or truncated file should not abort a training run.
            metrics = {}

    payload = dict(payload)
    payload["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    metrics[section] = payload
    metrics["generated_at"] = payload["generated_at"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved published metrics for '{section}' to: {path}")
    return metrics


def _fetch_datastore(resource_id: str, limit: int = 1000) -> list:
    url = f"{DATASTORE_URL}?resource_id={resource_id}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "hdb-price-predictor/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload.get("success"):
        raise RuntimeError(f"data.gov.sg returned an unsuccessful response for {resource_id}")
    return payload["result"]["records"]


def quarter_key(year: int, month: int) -> str:
    """'2024', 7 -> '2024-Q3'."""
    return f"{int(year)}-Q{(int(month) - 1) // 3 + 1}"


def _quarter_sort_key(q: str):
    year, quarter = q.split("-Q")
    return (int(year), int(quarter))


def fetch_hdb_index(cache_path: str) -> dict:
    """
    Returns {'2024-Q3': 197.9, ...} for the official HDB Resale Price Index.

    The download is cached so a retrain works offline; a stale cache is far
    better than aborting a training run.
    """
    try:
        records = _fetch_datastore(HDB_RPI_RESOURCE)
        index = {r["quarter"]: float(r["index"]) for r in records if r.get("index")}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        print(f"Fetched HDB Resale Price Index: {len(index)} quarters "
              f"({min(index, key=_quarter_sort_key)} to {max(index, key=_quarter_sort_key)}).")
        return index
    except Exception as exc:
        print(f"[!] Could not fetch HDB RPI ({exc}).")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            print(f"    Using cached copy: {len(index)} quarters.")
            return index
        print("    No cache available; falling back to a trend fitted from transactions.")
        return {}


def fetch_ura_index(cache_path: str) -> dict:
    """
    Returns {'CCR': {'2024-Q3': 148.2, ...}, 'RCR': {...}, 'OCR': {...}} for the
    URA non-landed property price index.
    """
    try:
        records = _fetch_datastore(URA_PPI_RESOURCE)
        index = {}
        for r in records:
            code = SEGMENT_CODES.get(r.get("market_segment", "").strip())
            if not code or not r.get("price_index"):
                continue
            index.setdefault(code, {})[r["quarter"]] = float(r["price_index"])
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        spans = ", ".join(f"{k}:{len(v)}" for k, v in sorted(index.items()))
        print(f"Fetched URA Property Price Index by segment ({spans} quarters).")
        return index
    except Exception as exc:
        print(f"[!] Could not fetch URA PPI ({exc}).")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            print(f"    Using cached copy.")
            return index
        print("    No cache available; falling back to a trend fitted from transactions.")
        return {}


def index_growth_rate(series: dict, years: float = 5.0) -> float:
    """
    Annualised growth over the last `years` of an index, used to project the
    index forward past its final published quarter.

    Measured over several years so one soft quarter does not set the long-run
    forecast. Clamped to a sane band: an index can fall, but compounding a
    steep decline (or a boom) out six years produces nonsense.
    """
    if len(series) < 8:
        return 0.025
    quarters = sorted(series, key=_quarter_sort_key)
    span = min(int(years * 4), len(quarters) - 1)
    start, end = series[quarters[-1 - span]], series[quarters[-1]]
    if start <= 0:
        return 0.025
    rate = (end / start) ** (4.0 / span) - 1.0
    return float(max(-0.02, min(0.08, rate)))

def load_cached_index(cache_path: str) -> dict:
    """
    Reads an index written by the fetch stage, without touching the network.

    Trainers use this so stage 3 stays offline and `--skip-fetch` genuinely
    skips every download. Returns {} when the cache is missing, which callers
    treat as "fall back to a trend fitted from the transactions".
    """
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

def sync_metrics_fallback(app_js_path: str, metrics: dict) -> bool:
    """
    Writes the published metrics into app.js as DEFAULT_METRICS.

    Opened straight from disk (file://), the browser blocks fetch(), so the page
    never loads models/metrics.json and every figure in the About panel renders
    as its "--" placeholder. DEFAULT_SCALER is inlined for the same reason.

    Regenerating the literal here means the offline copy cannot drift from the
    model that was actually trained.
    """
    if not os.path.exists(app_js_path):
        print(f"[!] {app_js_path} not found; skipping offline metrics sync.")
        return False

    with open(app_js_path, "r", encoding="utf-8") as f:
        source = f.read()

    marker = "const DEFAULT_METRICS = "
    start = source.find(marker)
    terminator = chr(10) + "};"
    end = source.find(terminator, start) if start != -1 else -1
    if start == -1 or end == -1:
        print("[!] DEFAULT_METRICS literal not found in app.js; "
              "skipping offline metrics sync.")
        return False

    body = json.dumps(metrics, indent=2, sort_keys=True).rstrip()
    if body.endswith("}"):
        body = body[:-1].rstrip()

    updated = source[:start] + marker + body + source[end:]
    with open(app_js_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"Synced offline metrics fallback into {os.path.basename(app_js_path)}")
    return True

# ---------------------------------------------------------------------------
# GeoJSON helpers
# ---------------------------------------------------------------------------


def feature_attributes(feature: dict) -> dict:
    """
    Returns a GeoJSON feature's attributes regardless of how the export encodes
    them.

    Data.gov.sg serves its layers two ways depending on vintage: older KML
    conversions bury the fields in an HTML table inside `Description`, while
    current GeoJSON downloads expose them as plain properties. Reading only one
    form fails silently against the other -- it matches nothing and writes an
    empty lookup while reporting success -- so both are handled here.
    """
    props = feature.get("properties", {}) or {}
    description = props.get("Description", "")
    if description:
        parsed = dict(re.findall(r"<th>(.*?)</th>\s*<td>(.*?)</td>", description))
        if parsed:
            return parsed
    return {k: ("" if v is None else str(v)) for k, v in props.items()}


def polygon_centroid(geometry: dict):
    """
    Average vertex of a geometry's outer ring, as (lon, lat).

    Not a true polygon centroid, but for a building footprint the difference is
    about 7 m at the median and 24 m at the 95th percentile -- inside the
    building either way, and far below the resolution any consumer here needs.
    Keeps the pipeline free of shapely.
    """
    gtype = geometry.get("type")
    if gtype == "Polygon":
        ring = geometry["coordinates"][0]
    elif gtype == "MultiPolygon":
        ring = geometry["coordinates"][0][0]
    elif gtype == "Point":
        c = geometry["coordinates"]
        return float(c[0]), float(c[1])
    else:
        return None
    if not ring:
        return None
    return (sum(p[0] for p in ring) / len(ring),
            sum(p[1] for p in ring) / len(ring))


# ---------------------------------------------------------------------------
# Market trend
# ---------------------------------------------------------------------------


def compute_market_trend(prices, areas, years) -> dict:
    """
    Measures the historical annual price trend so the front-end can extrapolate
    past the training window.

    XGBoost cannot predict beyond the sale_year range it saw during training --
    later years all fall into the same terminal leaf, producing a flat or even
    declining forecast. A log-linear regression through the median unit price
    per year gives the implied CAGR, which the app applies as an explicit
    multiplier instead.

    The most recent year is usually partial (data pulled mid-year), so it is
    excluded from the fit whenever at least two complete years remain.

    Takes three equal-length sequences rather than a DataFrame so both pipelines
    can call it despite holding their prices, areas and years in different
    shapes. Returns latest_year, latest_complete_year and annual_growth.
    """
    import numpy as np
    import pandas as pd

    unit_price = pd.Series(np.asarray(prices, dtype=float)) / \
        pd.Series(np.asarray(areas, dtype=float))
    medians = unit_price.groupby(pd.Series(np.asarray(years))).median().sort_index()

    latest_year = int(medians.index.max())

    # Drop the trailing partial year if enough history remains to fit.
    complete = medians.iloc[:-1] if len(medians) > 2 else medians
    latest_complete_year = int(complete.index.max())

    if len(complete) >= 2:
        yrs = complete.index.to_numpy(dtype=float)
        slope = np.polyfit(yrs, np.log(complete.to_numpy(dtype=float)), 1)[0]
        annual_growth = float(np.exp(slope) - 1.0)
    else:
        annual_growth = DEFAULT_ANNUAL_GROWTH

    # Guard against absurd extrapolation from a thin or noisy sample.
    annual_growth = float(np.clip(annual_growth, 0.0, MAX_ANNUAL_GROWTH))

    return {
        "latest_year": latest_year,
        "latest_complete_year": latest_complete_year,
        "annual_growth": round(annual_growth, 6),
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def find_compatible_python(project_root: str):
    """
    Returns a Python executable that has the training dependencies installed.

    A runner may be started by an interpreter without xgboost -- a system Python,
    or a shell that has not activated the project venv -- so the training stage
    can be delegated to one that does.
    """
    candidates = [
        sys.executable,
        os.path.join(project_root, ".venv", "Scripts", "python.exe"),
        os.path.join(project_root, ".venv", "bin", "python"),
    ]
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            res = subprocess.run(
                [candidate, "-c", "import xgboost, sklearn, pandas; print('OK')"],
                capture_output=True, text=True, timeout=15)
            if "OK" in res.stdout:
                return candidate
        except Exception:
            continue
    return None
