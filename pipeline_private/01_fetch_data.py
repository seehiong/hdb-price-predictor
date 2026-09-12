#!/usr/bin/env python3
"""
01_fetch_data.py
Fetches private residential and condominium transaction records and dwelling unit metadata:
  1. URA Data Service API (service=PMI_Resi_Transaction, batches 1-4)
  2. URA Number of Dwelling Units GeoJSON (Data.gov.sg dataset d_be71daeab5930f96b90ad2857454d876)
  3. URA Property Price Index of Non-landed Properties by Locality, quarterly -
     the official CCR / RCR / OCR market level series used to project prices
     past the training data cutoff.
"""

import csv
import json
import os
import time
import urllib.request
import sys
import urllib.parse

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
DATA_DIR = os.path.join(PIPELINE_DIR, "data")
sys.path.insert(0, PROJECT_ROOT)
from pipeline_common import fetch_ura_index

# Cached alongside the model artifacts: the trainer reads it, and it keeps a
# retrain working when data.gov.sg is unreachable.
INDEX_JSON = os.path.join(PROJECT_ROOT, "models", "ura_price_index.json")
OUTPUT_CSV = os.path.join(DATA_DIR, "private_resi_transactions.csv")
DWELLING_UNITS_GEOJSON = os.path.join(DATA_DIR, "URANoofDwellingUnits.geojson")
DWELLING_UNITS_RESOURCE_ID = "d_be71daeab5930f96b90ad2857454d876"
BCA_MCST_RESOURCE_ID = "d_f988c57e16e99ad3a649aa04572efd1c"
BCA_MCST_XLSX = os.path.join(DATA_DIR, "MCSTinformation.xlsx")
BCA_MCST_JSON = os.path.join(DATA_DIR, "bca_mcst_registry.json")
POLL_DOWNLOAD_URL = ("https://api-open.data.gov.sg/v1/public/api/datasets/"
                     "{}/poll-download")

URA_TOKEN_URL = "https://www.ura.gov.sg/uraDataService/insertNewToken.action"
URA_DATA_URL = "https://www.ura.gov.sg/uraDataService/invokeUraDS"


def load_env_access_key() -> str:
    """Check environment variables or local .env files for URA_ACCESS_KEY."""
    if os.environ.get("URA_ACCESS_KEY"):
        return os.environ.get("URA_ACCESS_KEY").strip()

    # Check root .env or pipeline .env
    for env_path in [os.path.join(PROJECT_ROOT, ".env"), os.path.join(PIPELINE_DIR, ".env")]:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("URA_ACCESS_KEY=") or line.startswith("URA_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip("\"'")
                        if val:
                            return val
    return ""


def get_ura_token(access_key: str) -> str:
    """Requests a daily authentication token from the URA Data Service."""
    headers = {
        "AccessKey": access_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    req = urllib.request.Request(URA_TOKEN_URL, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "success" and data.get("Result"):
                return data["Result"]
            raise RuntimeError(f"URA token response unsuccessful: {data}")
    except Exception as e:
        raise RuntimeError(f"Failed to acquire URA token: {e}")


def fetch_ura_transactions(access_key: str, token: str) -> list:
    """Fetches all 4 batches of private residential transactions from URA."""
    all_rows = []
    headers = {
        "AccessKey": access_key,
        "Token": token,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    for batch in [1, 2, 3, 4]:
        print(f"  Fetching batch {batch}/4 from URA Data Service...")
        params = urllib.parse.urlencode({
            "service": "PMI_Resi_Transaction",
            "batch": batch
        })
        url = f"{URA_DATA_URL}?{params}"
        req = urllib.request.Request(url, headers=headers)

        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    result = payload.get("Result", [])
                    batch_records = 0
                    for proj in result:
                        project_name = proj.get("project", "").strip()
                        street_name = proj.get("street", "").strip()
                        market_segment = proj.get("marketSegment", "").strip()
                        x_coord = proj.get("x", "")
                        y_coord = proj.get("y", "")

                        for tx in proj.get("transaction", []):
                            row = {
                                "project": project_name,
                                "street": street_name,
                                "market_segment": market_segment,
                                "x": x_coord,
                                "y": y_coord,
                                "area": tx.get("area", ""),
                                "floor_range": tx.get("floorRange", ""),
                                "no_of_units": tx.get("noOfUnits", "1"),
                                "contract_date": tx.get("contractDate", ""),
                                "type_of_sale": tx.get("typeOfSale", ""),
                                "price": tx.get("price", ""),
                                "property_type": tx.get("propertyType", ""),
                                "district": tx.get("district", ""),
                                "type_of_area": tx.get("typeOfArea", ""),
                                "tenure": tx.get("tenure", ""),
                                "nett_price": tx.get("nettPrice", "")
                            }
                            all_rows.append(row)
                            batch_records += 1

                    print(f"    Batch {batch} complete: retrieved {batch_records:,} transactions ({len(result):,} projects).")
                    break
            except Exception as e:
                if attempt == 3:
                    print(f"    [Error] Batch {batch} failed after 3 attempts: {e}")
                else:
                    time.sleep(2 * attempt)
        time.sleep(1)

    return all_rows


def ensure_dwelling_units_geojson():
    """
    Downloads URA Number of Dwelling Units if it is not already present.

    02_build_postal_data reads it to resolve each private development to a
    postal code, which is how the web app turns a postal code the user types
    into a project name. It used to be a manual prerequisite announced by a
    print statement; it is a file dataset on Data.gov.sg, so it is fetched the
    same way as the other geospatial layers.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DWELLING_UNITS_GEOJSON) and \
            os.path.getsize(DWELLING_UNITS_GEOJSON) > 100_000:
        size_mb = os.path.getsize(DWELLING_UNITS_GEOJSON) / (1024 * 1024)
        print(f"[CACHE HIT] Dwelling Units GeoJSON present: {size_mb:.1f} MB")
        return

    print("Downloading URA Number of Dwelling Units...")
    try:
        poll = urllib.request.Request(
            POLL_DOWNLOAD_URL.format(DWELLING_UNITS_RESOURCE_ID),
            headers={"User-Agent": "hdb-price-predictor/1.0"})
        with urllib.request.urlopen(poll, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        url = (payload.get("data") or {}).get("url")
        if not url:
            raise RuntimeError(payload.get("errMsg") or "no download URL returned")

        req = urllib.request.Request(
            url, headers={"User-Agent": "hdb-price-predictor/1.0"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = resp.read()

        # Parse before writing so a truncated or error body never replaces a
        # good local copy.
        json.loads(data)
        with open(DWELLING_UNITS_GEOJSON, "wb") as f:
            f.write(data)
        print(f"Saved Dwelling Units GeoJSON: "
              f"{len(data) / (1024 * 1024):.1f} MB -> {DWELLING_UNITS_GEOJSON}")
    except Exception as exc:
        print(f"[!] Could not fetch Dwelling Units GeoJSON: {exc}")
        print("    02_build_postal_data will fall back to the committed "
              "private_nonlanded_postal_data.json.")


def ensure_bca_mcst_registry():
    """
    Downloads and compiles the BCA Management Corporation Strata Title (MCST) Registry
    from Data.gov.sg (dataset d_f988c57e16e99ad3a649aa04572efd1c).

    Provides the official MCST establishment / constitution date for private strata
    developments, enabling the interactive map to systematically display community
    formation dates without ad-hoc hardcoding.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(BCA_MCST_JSON) and os.path.getsize(BCA_MCST_JSON) > 10_000:
        size_kb = os.path.getsize(BCA_MCST_JSON) / 1024
        print(f"[CACHE HIT] BCA MCST Registry JSON present: {size_kb:.1f} KB")
        return

    # Download XLSX from data.gov.sg if not present
    if not os.path.exists(BCA_MCST_XLSX) or os.path.getsize(BCA_MCST_XLSX) < 10_000:
        print("Downloading BCA MCST information from Data.gov.sg...")
        try:
            poll = urllib.request.Request(
                POLL_DOWNLOAD_URL.format(BCA_MCST_RESOURCE_ID),
                headers={"User-Agent": "hdb-price-predictor/1.0"}
            )
            with urllib.request.urlopen(poll, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            url = (payload.get("data") or {}).get("url")
            if not url:
                raise RuntimeError(payload.get("errMsg") or "no download URL returned")

            req = urllib.request.Request(
                url, headers={"User-Agent": "hdb-price-predictor/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                xlsx_bytes = resp.read()

            with open(BCA_MCST_XLSX, "wb") as f:
                f.write(xlsx_bytes)
            print(f"Saved BCA MCST raw XLSX: {len(xlsx_bytes) / 1024:.1f} KB -> {BCA_MCST_XLSX}")
        except Exception as exc:
            print(f"[!] Could not download BCA MCST dataset: {exc}")
            return

    # Parse XLSX and compile bca_mcst_registry.json
    print("Compiling BCA MCST registry JSON...")
    try:
        import re
        import openpyxl

        wb = openpyxl.load_workbook(BCA_MCST_XLSX, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return

        header = [str(c).lower() if c else "" for c in rows[0]]
        name_idx = header.index("usr_devtname")
        date_idx = header.index("mc_form_date")
        mcno_idx = header.index("usr_mcno")
        loc_idx = header.index("devt_location")

        def norm_name(s):
            return re.sub(
                r"[^A-Z0-9]", "",
                str(s).upper()
                .replace("CONDOMINIUM", "")
                .replace("CONDO", "")
                .replace("RESIDENCES", "")
                .replace("RESIDENCE", "")
                .replace("APARTMENTS", "")
                .replace("APARTMENT", "")
            )

        registry = {}
        for r in rows[1:]:
            dname = str(r[name_idx] or "").strip()
            fdate = str(r[date_idx] or "").strip()
            mcno = str(r[mcno_idx] or "").strip()
            loc = str(r[loc_idx] or "").strip()
            if not dname:
                continue

            ym = re.search(r"(\d{4})", fdate)
            year = int(ym.group(1)) if ym else None

            pm = re.search(r"\b(\d{6})\b", loc)
            postal = pm.group(1) if pm else ""

            np = norm_name(dname)
            if np and np not in registry:
                registry[np] = {
                    "name": dname,
                    "mcno": mcno,
                    "date": fdate,
                    "year": year,
                    "postal": postal,
                }

        with open(BCA_MCST_JSON, "w", encoding="utf-8") as out:
            json.dump(registry, out, separators=(",", ":"))
        print(f"Compiled {len(registry):,} BCA MCST records to: {BCA_MCST_JSON}")
    except Exception as exc:
        print(f"[!] Error processing BCA MCST XLSX: {exc}")


def download_private_data():
    """Main data download entry point."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=== Step 1: Fetching Private Residential Data ===")

    ensure_dwelling_units_geojson()
    ensure_bca_mcst_registry()

    access_key = load_env_access_key()
    if not access_key:
        print("\n[!] No 'URA_ACCESS_KEY' found in environment or .env file.")
        print("To download the latest private transactions directly from URA:")
        print("  1. Sign up for a free URA Data Service key at https://developer.tech.gov.sg/ or https://www.ura.gov.sg/maps/api/")
        print("  2. Add URA_ACCESS_KEY=<your_key> to a .env file in the project root.\n")

        if os.path.exists(OUTPUT_CSV):
            print(f"Using existing cached transaction dataset: {OUTPUT_CSV}")
            return OUTPUT_CSV
        else:
            print("[Notice] No existing transactions CSV found. Awaiting URA_ACCESS_KEY or local CSV.")
            return None

    print(f"Acquiring daily URA token with Access Key ({access_key[:4]}****)...")
    token = get_ura_token(access_key)
    print("Token obtained successfully. Downloading transaction batches...")

    records = fetch_ura_transactions(access_key, token)
    if not records:
        print("Warning: No records retrieved.")
        return None

    fieldnames = [
        "project", "street", "market_segment", "x", "y",
        "area", "floor_range", "no_of_units", "contract_date",
        "type_of_sale", "price", "property_type", "district",
        "type_of_area", "tenure", "nett_price"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records):,} private transactions to: {OUTPUT_CSV}\n")
    return OUTPUT_CSV


def download_price_index():
    """
    Fetches the URA non-landed price index (CCR / RCR / OCR) and caches it.

    XGBoost cannot extrapolate: any sale_year past the training range lands in
    the same terminal leaf, so the model alone returns a flat forecast. The
    trainer derives a per-segment annual growth rate from this series and writes
    it into condo_scaler.json for the browser to apply beyond the cutoff.

    The segment split is the point: CCR, RCR and OCR grew roughly +20%, +33% and
    +46% between 2021-Q1 and 2026-Q2, so one blended rate badly overstates
    prime-district forecasts.
    """
    print("=== Fetching URA Property Price Index ===")
    os.makedirs(os.path.dirname(INDEX_JSON), exist_ok=True)
    index = fetch_ura_index(INDEX_JSON)
    if not index:
        print("[!] Price index unavailable; training will fall back to a trend "
              "fitted from the transactions themselves.\n")
    else:
        print(f"Saved price index to: {INDEX_JSON}\n")
    return index


if __name__ == "__main__":
    download_private_data()
    download_price_index()
