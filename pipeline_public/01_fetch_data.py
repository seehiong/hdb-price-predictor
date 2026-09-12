#!/usr/bin/env python3
"""
01_fetch_data.py
Fetches the raw inputs for the public HDB pipeline from Data.gov.sg:

  1. HDB Resale Flat Prices (Jan 2017 onwards), with smart change detection,
     caching, and incremental fetching.
  2. HDB Resale Price Index (1Q2009 = 100), quarterly - the official market
     level series used to project prices past the training data cutoff.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from pipeline_common import fetch_hdb_index

RESOURCE_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
# HDB Property Information: block, street, storey count, year completed and
# dwelling-unit mix. Needed by the postal builder and the map generator.
PROPERTY_INFO_RESOURCE_ID = "d_17f5382f26140b1fdae0ba2ef6239d2f"
# This dataset has 24 columns, so the resale batch size trips a 413 on the
# datastore API. 5,000 rows per request is the largest that succeeds.
PROPERTY_INFO_BATCH_SIZE = 5000

# Geospatial layers. These are file downloads rather than datastore tables, so
# they come through the poll-download endpoint instead of datastore_search.
# Both describe HDB stock and are read only by this pipeline, so they live
# alongside the rest of its data.
GEOJSON_DATASETS = {
    "LTAMRTStationExitGEOJSON.geojson": (
        "d_b39d3a0871985372d7e1637193335da5",
        "LTA MRT Station Exits",
    ),
    "HDBExistingBuilding.geojson": (
        "d_16b157c52ed637edd6ba1232e026258d",
        "HDB Existing Building footprints",
    ),
}
POLL_DOWNLOAD_URL = ("https://api-open.data.gov.sg/v1/public/api/datasets/"
                     "{}/poll-download")
BASE_URL = "https://data.gov.sg/api/action/datastore_search"
BATCH_SIZE = 10000
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "resale_flat_prices_2017_present.csv")
PROPERTY_INFO_CSV = os.path.join(DATA_DIR, "hdb_property_info.csv")
METADATA_FILE = os.path.join(DATA_DIR, ".fetch_metadata.json")
# Cached alongside the model artifacts: the trainer reads it, and it keeps a
# retrain working when data.gov.sg is unreachable.
INDEX_JSON = os.path.join(PROJECT_ROOT, "models", "hdb_price_index.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

EXPECTED_COLS = [
    "month", "town", "flat_type", "block", "street_name",
    "storey_range", "floor_area_sqm", "flat_model",
    "lease_commence_date", "remaining_lease", "resale_price"
]


def fetch_batch(resource_id: str, limit: int, offset: int, retries: int = 5):
    params = urllib.parse.urlencode({
        "resource_id": resource_id,
        "limit": limit,
        "offset": offset,
        "sort": "_id asc"
    })
    url = f"{BASE_URL}?{params}"
    req = urllib.request.Request(url, headers=HEADERS)

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not data.get("success"):
                    raise RuntimeError(f"API call reported success=False: {data}")
                return data["result"]
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Failed to fetch offset {offset} after {retries} attempts: {e}")
            wait_time = attempt * 2
            print(f"  [Attempt {attempt}/{retries}] Error fetching offset {offset}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)


def count_csv_rows(csv_path: str) -> int:
    """Returns number of data rows in CSV (excluding header), or 0 if missing/empty."""
    if not os.path.exists(csv_path):
        return 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f) - 1
            return max(0, count)
    except Exception:
        return 0


def load_metadata() -> dict:
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_metadata(meta: dict):
    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"  [Warning] Failed to save metadata: {e}")


def download_all_resale_data(force: bool = False):
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=== Checking HDB Resale Flat Prices on Data.gov.sg ===")

    # Initial probe to check latest total count on Data.gov.sg
    probe = fetch_batch(RESOURCE_ID, limit=1, offset=0)
    remote_total = probe.get("total", 0)
    print(f"Latest records available on Data.gov.sg: {remote_total:,}")

    local_count = count_csv_rows(OUTPUT_CSV)
    meta = load_metadata()

    # 1. CACHE HIT: Local dataset is already identical to remote
    if not force and local_count == remote_total and remote_total > 0:
        print(f"\n[CACHE HIT] Local dataset is already completely up to date!")
        print(f"  - File: {OUTPUT_CSV}")
        print(f"  - Local rows:  {local_count:,}")
        print(f"  - Remote rows: {remote_total:,}")
        print("  - No new transactions published. Skipping download.\n")
        return OUTPUT_CSV

    # 2. INCREMENTAL FETCH: Local dataset exists and has valid subset
    if not force and 0 < local_count < remote_total:
        new_records_count = remote_total - local_count
        print(f"\n[INCREMENTAL UPDATE] Detected {new_records_count:,} new transactions!")
        print(f"  - Local rows:  {local_count:,}")
        print(f"  - Remote rows: {remote_total:,}")
        print(f"  - Fetching only new records from offset {local_count:,}...\n")

        new_records = []
        offset = local_count
        start_time = time.time()

        while offset < remote_total:
            batch_limit = min(BATCH_SIZE, remote_total - offset)
            print(f"  Fetching records {offset + 1:,} to {offset + batch_limit:,} of {remote_total:,}...")
            result = fetch_batch(RESOURCE_ID, limit=batch_limit, offset=offset)
            records = result.get("records", [])
            if not records:
                break
            new_records.extend(records)
            offset += len(records)
            time.sleep(0.2)

        elapsed = time.time() - start_time
        print(f"Downloaded {len(new_records):,} new records in {elapsed:.1f}s.")

        # Append new records to existing CSV
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EXPECTED_COLS, extrasaction="ignore")
            writer.writerows(new_records)

        updated_total = count_csv_rows(OUTPUT_CSV)
        print(f"Appended to: {OUTPUT_CSV}")
        print(f"Total dataset rows now: {updated_total:,}")

        save_metadata({
            "resource_id": RESOURCE_ID,
            "total_records": remote_total,
            "row_count": updated_total,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "up-to-date"
        })
        print("=== Incremental Update Complete ===\n")
        return OUTPUT_CSV

    # 3. FULL DOWNLOAD: File missing, corrupted, local_count > remote_total, or --force
    reason = "Force flag specified" if force else ("CSV missing" if local_count == 0 else "Local row count out of sync")
    print(f"\n[FULL DOWNLOAD] Initiating complete download ({reason})...")

    all_records = []
    offset = 0
    start_time = time.time()

    while offset < remote_total:
        batch_limit = min(BATCH_SIZE, remote_total - offset)
        print(f"  Fetching records {offset + 1:,} to {offset + batch_limit:,} of {remote_total:,}...")
        result = fetch_batch(RESOURCE_ID, limit=batch_limit, offset=offset)
        records = result.get("records", [])
        if not records:
            break
        all_records.extend(records)
        offset += len(records)
        time.sleep(0.2)

    elapsed = time.time() - start_time
    print(f"Downloaded {len(all_records):,} records in {elapsed:.1f}s.")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)

    min_month = min((r.get("month", "") for r in all_records if r.get("month")), default="N/A")
    max_month = max((r.get("month", "") for r in all_records if r.get("month")), default="N/A")

    save_metadata({
        "resource_id": RESOURCE_ID,
        "total_records": remote_total,
        "row_count": len(all_records),
        "date_range": f"{min_month} to {max_month}",
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "up-to-date"
    })

    print(f"Saved dataset to: {OUTPUT_CSV}")
    print(f"Total rows: {len(all_records):,}")
    print(f"Date range: {min_month} to {max_month}")
    print("=== Fetch Complete ===\n")
    return OUTPUT_CSV


def download_property_info():
    """
    Fetches HDB Property Information (block, street, storey count, year
    completed, dwelling-unit mix) from Data.gov.sg.

    Two later stages need it and neither can produce it: 02_build_postal_data
    uses it to attach postal codes, and 04_build_map reads storey counts and
    unit mixes for the map popups. Without it the postal builder silently falls
    back to the committed lookup and the map generator fails outright, so it is
    fetched here rather than left as an undocumented manual prerequisite.
    """
    print("=== Fetching HDB Property Information ===")
    os.makedirs(DATA_DIR, exist_ok=True)

    probe = fetch_batch(PROPERTY_INFO_RESOURCE_ID, limit=1, offset=0)
    total = probe.get("total", 0)
    if not total:
        print("[!] Could not determine record count; skipping property info.\n")
        return None

    if os.path.exists(PROPERTY_INFO_CSV):
        local = count_csv_rows(PROPERTY_INFO_CSV)
        if local == total:
            print(f"[CACHE HIT] {local:,} records already downloaded.\n")
            return PROPERTY_INFO_CSV

    print(f"Downloading {total:,} property records...")
    records = []
    offset = 0
    while offset < total:
        batch = fetch_batch(PROPERTY_INFO_RESOURCE_ID,
                            limit=min(PROPERTY_INFO_BATCH_SIZE, total - offset),
                            offset=offset)
        rows = batch.get("records", [])
        if not rows:
            break
        records.extend(rows)
        offset += len(rows)
        print(f"  {len(records):,}/{total:,}")

    if not records:
        print("[!] No property records returned; skipping.\n")
        return None

    # _id is a datastore artefact, not part of the dataset.
    fields = [f for f in records[0].keys() if f != "_id"]
    with open(PROPERTY_INFO_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records):,} records to: {PROPERTY_INFO_CSV}\n")
    return PROPERTY_INFO_CSV


def download_geospatial_layers():
    """
    Downloads the GeoJSON layers the map and MRT-proximity stages are built from.

    Neither layer is committed -- together they are around 57 MB -- and without
    them 02b and 02c fall back to the lookups already in the repository. That
    works, but it means those stages cannot be rebuilt from scratch, so they are
    fetched here instead of being left as a manual prerequisite.

    Data.gov.sg serves file datasets through a poll-download endpoint that hands
    back a signed S3 URL, rather than through the datastore API used for tables.
    """
    print("=== Fetching geospatial layers ===")
    target_dir = DATA_DIR
    os.makedirs(target_dir, exist_ok=True)

    for filename, (resource_id, label) in GEOJSON_DATASETS.items():
        path = os.path.join(target_dir, filename)
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [CACHE HIT] {label}: {size_mb:.1f} MB already present.")
            continue

        try:
            poll = urllib.request.Request(
                POLL_DOWNLOAD_URL.format(resource_id), headers=HEADERS)
            with urllib.request.urlopen(poll, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            url = (payload.get("data") or {}).get("url")
            if not url:
                raise RuntimeError(payload.get("errMsg") or "no download URL returned")

            print(f"  Downloading {label}...")
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()

            # Write only after a successful parse, so a truncated or error
            # response never replaces a good local copy.
            json.loads(data)
            with open(path, "wb") as f:
                f.write(data)
            print(f"  Saved {label}: {len(data) / (1024 * 1024):.1f} MB -> {filename}")
        except Exception as exc:
            print(f"  [!] Could not fetch {label}: {exc}")
            print(f"      02b/02c will fall back to the committed lookups.")

    print()


def download_price_index():
    """
    Fetches the official HDB Resale Price Index and caches it under models/.

    XGBoost cannot extrapolate: any sale_year past the training range lands in
    the same terminal leaf, so the model alone returns a flat forecast. The
    trainer reads this series, derives an annual growth rate from it, and writes
    that rate into hdb_scaler.json for the browser to apply beyond the cutoff.
    """
    print("=== Fetching HDB Resale Price Index ===")
    os.makedirs(os.path.dirname(INDEX_JSON), exist_ok=True)
    index = fetch_hdb_index(INDEX_JSON)
    if not index:
        print("[!] Price index unavailable; training will fall back to a trend "
              "fitted from the transactions themselves.\n")
    else:
        print(f"Saved price index to: {INDEX_JSON}\n")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch HDB resale price data from Data.gov.sg with caching.")
    parser.add_argument("--force", action="store_true", help="Force complete redownload even if cached")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip refreshing the HDB Resale Price Index")
    args = parser.parse_args()

    download_all_resale_data(force=args.force)
    download_property_info()
    download_geospatial_layers()
    if not args.skip_index:
        download_price_index()
