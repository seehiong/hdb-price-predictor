#!/usr/bin/env python3
"""
02b_build_mrt_proximity.py
Computes walking-relevant distance from every HDB block to the nearest MRT
station exit, and writes a (block, town) -> distance lookup used by both the
trainer and the web app.

Proximity to an MRT exit is one of the strongest location signals in HDB resale
pricing: median unit price falls steadily from roughly S$5,530/sqm within 250 m
to S$4,960/sqm at 1-1.5 km. Unlike a block-level average price, distance is a
stable geographic fact, so it stays valid as the market moves.

Every one of the 563 station exits is compared against every block and the
nearest is kept, so blocks served by more than one station are handled without
any spatial database - it is a 12,847 x 563 scan that runs in seconds.

Known limitation: the trainer joins on (block, town) because that is all the
resale CSV provides, and block numbers repeat within a town. About 596 buildings
share a key with another building - "Block 12, Bukit Merah" is four separate
blocks between 0.13 km and 1.24 km from a station - and those keys carry the
mean of their members, blurring the feature for roughly 7% of transactions.
Resolving it needs a postal-to-street mapping that only an external geocoder can
supply. It was measured and judged not worth the dependency: the feature is
worth S$2,745 of MAE even with the blur (7.22% vs 7.54% MAPE), and the remaining
93% of rows are already exact, so perfect disambiguation would recover only a
fraction of that. The web app is unaffected either way - it looks up by postal
code, which is unique.

Sources:
  - LTAMRTStationExitGEOJSON.geojson (LTA, via Data.gov.sg)
  - HDBExistingBuilding.geojson      (HDB, via Data.gov.sg)
"""

import json
import math
import os
import sys
import re

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PROJECT_ROOT)
from pipeline_common import feature_attributes, polygon_centroid
DATA_DIR = os.path.join(PIPELINE_DIR, "data")

# Geometry inputs live in the private pipeline's data folder (shared geodata).
MRT_GEOJSON = os.path.join(DATA_DIR, "LTAMRTStationExitGEOJSON.geojson")
HDB_BUILDING_GEOJSON = os.path.join(DATA_DIR, "HDBExistingBuilding.geojson")
POSTAL_JSON = os.path.join(DATA_DIR, "hdb_postal_data.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "hdb_mrt_proximity.json")

# Singapore sits near 1.35 N; these convert degrees to km accurately enough for
# ranking nearby stations (sub-metre error at this scale).
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32


def load_mrt_exits(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    exits = []
    for feature in data.get("features", []):
        point = polygon_centroid(feature.get("geometry", {}))
        if point:
            exits.append(point)
    print(f"  Loaded {len(exits):,} MRT station exits.")
    return exits


def nearest_exit_km(lon: float, lat: float, exits) -> float:
    lon_scale = KM_PER_DEG_LON * math.cos(math.radians(lat))
    best = float("inf")
    for ex_lon, ex_lat in exits:
        dx = (ex_lon - lon) * lon_scale
        dy = (ex_lat - lat) * KM_PER_DEG_LAT
        d = dx * dx + dy * dy
        if d < best:
            best = d
    return math.sqrt(best)


def build_mrt_proximity():
    print("=== Building HDB -> MRT Proximity Lookup ===")

    for path in (MRT_GEOJSON, HDB_BUILDING_GEOJSON):
        if not os.path.exists(path):
            print(f"[!] Missing required geodata: {path}")
            print("    Skipping MRT proximity build; the model will train without it.")
            return None

    exits = load_mrt_exits(MRT_GEOJSON)
    if not exits:
        print("[!] No MRT exits parsed; aborting.")
        return None

    postal_to_town = {}
    if os.path.exists(POSTAL_JSON):
        with open(POSTAL_JSON, "r", encoding="utf-8") as f:
            for postal, entry in json.load(f).items():
                record = entry[0] if isinstance(entry, list) and entry else entry
                if isinstance(record, dict) and record.get("town"):
                    postal_to_town[postal] = record["town"].strip().upper()
    print(f"  Loaded {len(postal_to_town):,} postal -> town mappings.")

    with open(HDB_BUILDING_GEOJSON, "r", encoding="utf-8") as f:
        buildings = json.load(f)

    # Two views of the same distances:
    #   by_block  keyed "BLK|TOWN" - what the resale CSV can join on at train time
    #   by_postal keyed on the 6-digit code - what the web app has from user input
    lookup = {}
    by_postal = {}
    for feature in buildings.get("features", []):
        attrs = feature_attributes(feature)
        blk = attrs.get("BLK_NO", "").strip().upper()
        postal = attrs.get("POSTAL_COD", "").strip()
        town = postal_to_town.get(postal)
        point = polygon_centroid(feature.get("geometry", {}))
        if not point:
            continue
        km = nearest_exit_km(point[0], point[1], exits)
        if len(postal) == 6:
            by_postal[postal] = round(km, 4)
        if blk and town:
            lookup.setdefault(f"{blk}|{town}", []).append(km)

    flat = {key: round(sum(v) / len(v), 4) for key, v in lookup.items()}
    print(f"  Computed MRT distance for {len(flat):,} (block, town) pairs "
          f"and {len(by_postal):,} postal codes.")

    if flat:
        values = sorted(flat.values())
        print(f"  Median {values[len(values) // 2]:.3f} km | "
              f"min {values[0]:.3f} | max {values[-1]:.3f}")

    payload = {
        "unit": "km",
        "description": "Straight-line distance from HDB block centroid to nearest MRT exit",
        "median_km": round(sorted(flat.values())[len(flat) // 2], 4) if flat else None,
        "distances": flat,
        "by_postal": by_postal,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Saved MRT proximity lookup to: {OUTPUT_JSON}\n")
    return payload


if __name__ == "__main__":
    build_mrt_proximity()
