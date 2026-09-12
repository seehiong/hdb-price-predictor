#!/usr/bin/env python3
"""
02_build_postal_data.py
Generates an updated postal_data.json mapping 6-digit Singapore postal codes
to HDB town and lease commencement year.
"""

import csv
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROP_INFO_CSV = os.path.join(DATA_DIR, "hdb_property_info.csv")
RESALE_CSV = os.path.join(DATA_DIR, "resale_flat_prices_2017_present.csv")
EXISTING_POSTAL_JSON = os.path.join(DATA_DIR, "hdb_postal_data.json")
OUTPUT_POSTAL_JSON = os.path.join(DATA_DIR, "hdb_postal_data.json")

def build_postal_data():
    print("=== Building Postal Code Lookup Database (hdb_postal_data.json) ===")
    
    # 1. Start with existing postal data as baseline
    lookup = {}
    baseline_path = EXISTING_POSTAL_JSON
    if not os.path.exists(baseline_path):
        legacy_path = os.path.join(PROJECT_ROOT, "postal_data.json")
        if os.path.exists(legacy_path):
            baseline_path = legacy_path

    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                lookup = json.load(f)
            print(f"Loaded {len(lookup):,} baseline postal codes from {os.path.basename(baseline_path)}")
        except Exception as e:
            print(f"Warning: Could not read baseline postal data: {e}")
            lookup = {}

    # 2. Extract (block, street) -> (town, lease_commence_date) from resale data
    resale_map = {}
    csv_to_use = RESALE_CSV if os.path.exists(RESALE_CSV) else None

    if csv_to_use and os.path.exists(csv_to_use):
        print(f"Reading resale records from: {csv_to_use}")
        with open(csv_to_use, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                blk = row.get("block", "").strip().upper()
                street = row.get("street_name", "").strip().upper()
                town = row.get("town", "").strip().upper()
                lcd = row.get("lease_commence_date", "").strip()
                if blk and street and town and lcd:
                    resale_map[(blk, street)] = (town, lcd)
        print(f"Mapped {len(resale_map):,} unique (block, street) pairs from resale transactions.")

    # 3. Read hdb_property_info.csv to link postal codes
    if os.path.exists(PROP_INFO_CSV):
        print(f"Reading property info from: {PROP_INFO_CSV}")
        added_count = 0
        with open(PROP_INFO_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                postal = str(row.get("postal", "")).strip().zfill(6)
                if not postal or postal == "000000" or not postal.isdigit():
                    continue

                blk = row.get("blk_no", "").strip().upper()
                street1 = row.get("canonical_street", "").strip().upper() if row.get("canonical_street") else ""
                street2 = row.get("street", "").strip().upper() if row.get("street") else ""

                town_info = None
                if (blk, street1) in resale_map:
                    town_info = resale_map[(blk, street1)]
                elif (blk, street2) in resale_map:
                    town_info = resale_map[(blk, street2)]

                if town_info:
                    town, lcd = town_info
                    if postal not in lookup:
                        added_count += 1
                    lookup[postal] = [{"town": town, "lease_commence_date": str(lcd)}]
                elif postal not in lookup and row.get("year_completed"):
                    # Fallback lease commencement year from year_completed if town is already known
                    pass

        print(f"Added/Updated {added_count:,} new postal codes from property info.")
    else:
        print(f"Warning: {PROP_INFO_CSV} not found, using baseline.")

    # 4. Sort and write to file
    sorted_lookup = {k: lookup[k] for k in sorted(lookup.keys())}
    with open(OUTPUT_POSTAL_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted_lookup, f, indent=2)

    print(f"Total postal codes in output {OUTPUT_POSTAL_JSON}: {len(sorted_lookup):,}")
    print("=== Postal Data Generation Complete ===\n")
    return OUTPUT_POSTAL_JSON

if __name__ == "__main__":
    build_postal_data()
