#!/usr/bin/env python3
"""
02c_enrich_property_info.py
Attaches postal codes and coordinates to the HDB Property Information table.

Data.gov.sg publishes HDB Property Information with 24 columns and no location
at all -- no postal code, no coordinates. Two things downstream need location:

  - 04_build_map.js plots one pin per block and reads longitude/latitude
  - 02b_build_mrt_proximity keys distances by postal code

The building footprints in HDBExistingBuilding.geojson carry both a block number
and a postal code, so the two can be joined on (block, street-derived town) and
the footprint centroid used as the block's position. That turns the published
24-column file into the 34-column form the rest of the pipeline expects.

Centroids are the average of the footprint's outer ring rather than a true
polygon centroid. Compared against a reference set computed with shapely the
median difference is ~7 m and the 95th percentile ~24 m -- well inside a single
building, and far below the resolution at which either consumer cares. Doing it
this way keeps the pipeline on the standard library: no shapely, no pyproj, no
geospatial database.

Coordinates are WGS84 (EPSG:4326). The x/y columns are SVY21 (EPSG:3414),
Singapore's national grid, derived here with a direct implementation of the
projection rather than a dependency.
"""

import csv
import json
import math
import os
import sys
import re
import time
import urllib.error
import urllib.request

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PROJECT_ROOT)
from pipeline_common import feature_attributes, polygon_centroid
DATA_DIR = os.path.join(PIPELINE_DIR, "data")

PROPERTY_CSV = os.path.join(DATA_DIR, "hdb_property_info.csv")
BUILDING_GEOJSON = os.path.join(DATA_DIR, "HDBExistingBuilding.geojson")
RESALE_CSV = os.path.join(DATA_DIR, "resale_flat_prices_2017_present.csv")
ONEMAP_CACHE = os.path.join(DATA_DIR, ".onemap_postal_cache.json")

# OneMap resolves a postal code to its authoritative block, street and
# coordinates. It is the only source here that can say which of the many
# "block 32"s a postal code belongs to.
#
# This endpoint is public: no API key, no token, no account. Verified by
# calling it with every ONEMAP_* variable stripped from the environment.
# OneMap does gate some endpoints behind a Bearer token -- /public/revgeocode
# returns 401 without one -- but /common/elastic/search is not among them, so
# nothing here needs credentials and none are read.
ONEMAP_SEARCH = ("https://www.onemap.gov.sg/api/common/elastic/search"
                 "?searchVal={}&returnGeom=Y&getAddrDetails=Y&pageNum=1")
ONEMAP_DELAY_S = 1.2
ONEMAP_MAX_RETRIES = 3

# HDB stores an abbreviated town code in bldg_contract_town; the resale data
# uses full names.
TOWN_CODES = {
    "AMK": "ANG MO KIO", "BB": "BUKIT BATOK", "BD": "BEDOK", "BH": "BISHAN",
    "BM": "BUKIT MERAH", "BP": "BUKIT PANJANG", "BT": "BUKIT TIMAH",
    "CCK": "CHOA CHU KANG", "CL": "CLEMENTI", "CT": "CENTRAL AREA",
    "GL": "GEYLANG", "HG": "HOUGANG", "JE": "JURONG EAST", "JW": "JURONG WEST",
    "KWN": "KALLANG/WHAMPOA", "MP": "MARINE PARADE", "PG": "PUNGGOL",
    "PRC": "PASIR RIS", "QT": "QUEENSTOWN", "SB": "SEMBAWANG", "SGN": "SERANGOON",
    "SK": "SENGKANG", "TAP": "TAMPINES", "TG": "TENGAH", "TP": "TOA PAYOH",
    "WL": "WOODLANDS", "YS": "YISHUN",
}

# Columns appended to the published 24. Order matters: 04_build_map.js reads
# longitude and latitude positionally at indices 26 and 27.
ENRICHED_COLUMNS = [
    "postal", "full_address", "lon", "lat", "x", "y",
    "geom_4326", "geom_3414", "updated_at", "canonical_street",
]

# SVY21 / EPSG:3414 parameters (Singapore national grid).
SVY21_A = 6378137.0
SVY21_F = 1 / 298.257223563
SVY21_ORIGIN_LAT = 1.366666
SVY21_ORIGIN_LON = 103.833333
SVY21_FALSE_N = 38744.572
SVY21_FALSE_E = 28001.642
SVY21_K = 1.0

# HDB abbreviates street names inconsistently between datasets; normalising both
# sides lets the resale CSV join against this table.
STREET_ABBREVIATIONS = {
    "ROAD": "RD", "STREET": "ST", "AVENUE": "AVE", "CLOSE": "CL",
    "CRESCENT": "CRES", "DRIVE": "DR", "PLACE": "PL", "TERRACE": "TER",
    "JALAN": "JLN", "LORONG": "LOR", "BUKIT": "BT", "TANJONG": "TG",
    "KAMPONG": "KG", "UPPER": "UPP", "NORTH": "NTH", "SOUTH": "STH",
    "CENTRAL": "CTRL", "GARDENS": "GDNS", "HEIGHTS": "HTS", "PARK": "PK",
    "MARKET": "MKT", "COMMONWEALTH": "C'WEALTH", "SAINT": "ST.",
}


def normalise_street(name: str) -> str:
    """Collapses a street name to the abbreviated form HDB datasets use."""
    words = str(name).strip().upper().split()
    return " ".join(STREET_ABBREVIATIONS.get(w, w) for w in words)


def latlon_to_svy21(lat: float, lon: float):
    """
    Converts WGS84 degrees to SVY21 easting/northing.

    Transverse Mercator on the WGS84 ellipsoid with Singapore's grid origin.
    Implemented directly so the pipeline does not need pyproj for what is a
    handful of arithmetic per row.
    """
    b = SVY21_A * (1 - SVY21_F)
    e2 = (2 * SVY21_F) - (SVY21_F * SVY21_F)
    e4, e6 = e2 * e2, e2 * e2 * e2
    a1 = SVY21_A * (1 + (-e2 / 4) + (-3 * e4 / 64) + (-5 * e6 / 256))
    a2 = SVY21_A * ((-3 * e2 / 8) + (-3 * e4 / 32) + (-45 * e6 / 1024))
    a3 = SVY21_A * ((15 * e4 / 256) + (45 * e6 / 1024))
    a4 = SVY21_A * (-35 * e6 / 3072)

    def meridian(rad):
        return (a1 * rad + a2 * math.sin(2 * rad) + a3 * math.sin(4 * rad)
                + a4 * math.sin(6 * rad))

    lat_r = math.radians(lat)
    origin_r = math.radians(SVY21_ORIGIN_LAT)
    m = meridian(lat_r)
    m0 = meridian(origin_r)

    sin_lat, cos_lat, tan_lat = math.sin(lat_r), math.cos(lat_r), math.tan(lat_r)
    rho = SVY21_A * (1 - e2) / ((1 - e2 * sin_lat ** 2) ** 1.5)
    v = SVY21_A / math.sqrt(1 - e2 * sin_lat ** 2)
    psi = v / rho
    w = math.radians(lon - SVY21_ORIGIN_LON)
    t = tan_lat

    north = SVY21_FALSE_N + SVY21_K * (
        (m - m0)
        + (w ** 2 / 2) * v * sin_lat * cos_lat
        + (w ** 4 / 24) * v * sin_lat * cos_lat ** 3 * (4 * psi ** 2 + psi - t ** 2)
    )
    east = SVY21_FALSE_E + SVY21_K * (
        w * v * cos_lat
        + (w ** 3 / 6) * v * cos_lat ** 3 * (psi - t ** 2)
    )
    return east, north


def load_block_locations(path: str) -> dict:
    """Builds {block_no: [(postal, lon, lat), ...]} from the building footprints."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    locations = {}
    for feature in data.get("features", []):
        attrs = feature_attributes(feature)
        blk = attrs.get("BLK_NO", "").strip().upper()
        postal = attrs.get("POSTAL_COD", "").strip()
        point = polygon_centroid(feature.get("geometry", {}))
        if not blk or len(postal) != 6 or not point:
            continue
        locations.setdefault(blk, []).append((postal, point[0], point[1]))

    print(f"  Loaded footprints for {len(locations):,} distinct block numbers "
          f"({sum(len(v) for v in locations.values()):,} buildings).")
    return locations


def resolve_postals_via_onemap(postals, cache_path: str = ONEMAP_CACHE) -> dict:
    """
    Returns {postal: {"blk", "street", "lon", "lat"}} from OneMap.

    The building footprints give a block number and a postal code but no
    street, and block numbers are not unique -- block 32 exists in Chinatown,
    Eunos and elsewhere. Choosing a footprint by block alone therefore picks an
    arbitrary building, which is how 32 NEW MKT RD ended up plotted 7.4 km away
    in Bedok. OneMap resolves a postal code to its actual street, which settles
    it.

    Successful lookups are cached; failures are not, so a rate-limited run can
    be resumed by running again.
    """
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}

    todo = sorted({p for p in postals if p not in cache})
    if not todo:
        print(f"  OneMap: all {len(set(postals)):,} postal codes already cached.")
        return cache

    print(f"  OneMap: resolving {len(todo):,} postal codes "
          f"(~{len(todo) * ONEMAP_DELAY_S / 60:.0f} min, cached for next time)...")
    failures = 0
    for i, postal in enumerate(todo, 1):
        record = None
        for attempt in range(ONEMAP_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    ONEMAP_SEARCH.format(postal),
                    headers={"User-Agent": "hdb-price-predictor/1.0"})
                with urllib.request.urlopen(req, timeout=25) as resp:
                    results = json.loads(resp.read().decode("utf-8")).get("results") or []
                if results:
                    top = results[0]
                    record = {
                        "blk": str(top.get("BLK_NO", "")).strip().upper(),
                        "street": str(top.get("ROAD_NAME", "")).strip().upper(),
                        "lon": float(top.get("LONGITUDE", 0) or 0),
                        "lat": float(top.get("LATITUDE", 0) or 0),
                    }
                else:
                    record = {}
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    time.sleep(ONEMAP_DELAY_S * (2 ** attempt))
                    continue
                break
            except Exception:
                break

        if record is None:
            failures += 1
        else:
            cache[postal] = record

        if i % 250 == 0 or i == len(todo):
            print(f"    {i:,}/{len(todo):,} ({failures} unresolved)")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        time.sleep(ONEMAP_DELAY_S)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    if failures:
        print(f"  [!] {failures} lookups did not complete; run again to retry them.")
    return cache


def load_sector_towns(resale_csv: str, locations: dict) -> dict:
    """
    Learns {postal sector: town} from the resale transactions.

    A block number is not unique: block 328 exists in twelve different towns.
    Joining the property table to a footprint on block alone therefore picks an
    arbitrary building, and the postal code that comes with it can belong to a
    completely different part of the island. Postal sectors do not have that
    problem -- the first two digits place a postal code geographically -- so the
    resale data is used to learn which town each sector belongs to, and that
    decides which candidate footprint is the right one.
    """
    if not os.path.exists(resale_csv):
        return {}

    # (block, street) -> town, straight from the transactions.
    block_street_town = {}
    with open(resale_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            blk = str(row.get("block", "")).strip().upper()
            street = str(row.get("street_name", "")).strip().upper()
            town = str(row.get("town", "")).strip().upper()
            if blk and street and town:
                block_street_town[(blk, street)] = town

    # Blocks that appear in exactly one town let us attribute their postal
    # sector with confidence; ambiguous blocks are skipped.
    town_by_block = {}
    for (blk, _street), town in block_street_town.items():
        town_by_block.setdefault(blk, set()).add(town)

    votes = {}
    for blk, candidates in locations.items():
        towns = town_by_block.get(blk)
        if not towns or len(towns) != 1 or len(candidates) != 1:
            continue
        sector = candidates[0][0][:2]
        votes.setdefault(sector, {})
        town = next(iter(towns))
        votes[sector][town] = votes[sector].get(town, 0) + 1

    sector_town = {sec: max(counts, key=counts.get) for sec, counts in votes.items()}
    print(f"  Learned {len(sector_town)} postal sector to town mappings.")
    return sector_town


def enrich_property_info():
    print("=== Enriching HDB Property Information with coordinates ===")

    if not os.path.exists(PROPERTY_CSV):
        print(f"[!] {PROPERTY_CSV} not found.")
        print("    Run 01_fetch_data.py first.")
        return None

    with open(PROPERTY_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[!] Property info file is empty; nothing to enrich.")
        return None

    base_columns = [c for c in rows[0].keys() if c not in ENRICHED_COLUMNS]
    if all(c in rows[0] for c in ("lon", "lat")):
        located = sum(1 for r in rows if r.get("lat") and r.get("lon"))
        if located:
            print(f"  Already enriched: {located:,}/{len(rows):,} rows have "
                  f"coordinates. Nothing to do.\n")
            return PROPERTY_CSV

    if not os.path.exists(BUILDING_GEOJSON):
        print(f"[!] Building footprints not found: {BUILDING_GEOJSON}")
        print("    Download HDBExistingBuilding.geojson from Data.gov.sg into")
        print("    pipeline_public/data/ to enable the map and postal lookups.")
        print("    Training and the web app work without it.\n")
        return None

    locations = load_block_locations(BUILDING_GEOJSON)

    # Resolve every candidate postal code to its real street via OneMap. This
    # is what makes the join correct: the footprints know a block number and a
    # postal code but not which street, and block numbers repeat across the
    # island, so only an address service can say which "block 32" a postal code
    # actually is.
    all_postals = [c[0] for cands in locations.values() for c in cands]
    onemap = resolve_postals_via_onemap(all_postals)

    # postal -> street, keyed the way the property table spells streets.
    street_by_postal = {
        pc: normalise_street(rec.get("street", ""))
        for pc, rec in onemap.items() if rec.get("street")
    }

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    matched = 0
    unresolved = 0
    used_postals = set()
    for row in rows:
        blk = str(row.get("blk_no", "")).strip().upper()
        street = str(row.get("street", "")).strip().upper()
        row["canonical_street"] = normalise_street(street)
        row["updated_at"] = stamp

        # Match on block AND street. Block alone is ambiguous -- block 32 is a
        # different building in Chinatown and in Eunos -- and picking the wrong
        # one puts the pin kilometres away and corrupts the postal lookup built
        # from this file.
        candidates = locations.get(blk, [])
        want = row["canonical_street"]

        chosen = next(
            (c for c in candidates
             if street_by_postal.get(c[0]) == want and c[0] not in used_postals),
            None)
        if chosen is None:
            chosen = next(
                (c for c in candidates if street_by_postal.get(c[0]) == want), None)
        if chosen is None and len(candidates) == 1 and not street_by_postal:
            # No OneMap data at all (offline first run): a block with a single
            # candidate is unambiguous, so it is still safe to use.
            chosen = candidates[0]
        if chosen is None:
            # Either the street did not match any candidate, or the block is
            # ambiguous and we cannot tell which building this row is. Leave it
            # unlocated rather than guessing: a missing coordinate is visibly
            # missing, a wrong one is not.
            unresolved += 1
        if not chosen:
            for col in ("postal", "full_address", "lon", "lat", "x", "y",
                        "geom_4326", "geom_3414"):
                row[col] = ""
            continue

        postal, lon, lat = chosen
        # OneMap's coordinates are the surveyed address point; prefer them over
        # our footprint centroid where we have them.
        fix = onemap.get(postal) or {}
        if fix.get('lat') and fix.get('lon'):
            lon, lat = fix['lon'], fix['lat']
        used_postals.add(postal)
        east, north = latlon_to_svy21(lat, lon)

        row["postal"] = postal
        row["full_address"] = f"{blk} {street} SINGAPORE {postal}"
        row["lon"] = f"{lon:.12f}"
        row["lat"] = f"{lat:.12f}"
        row["x"] = f"{east:.6f}"
        row["y"] = f"{north:.6f}"
        row["geom_4326"] = f"POINT ({lon:.12f} {lat:.12f})"
        row["geom_3414"] = f"POINT ({east:.6f} {north:.6f})"
        matched += 1

    with open(PROPERTY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_columns + ENRICHED_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    pct = matched / len(rows) * 100 if rows else 0
    print(f"  Located {matched:,}/{len(rows):,} blocks ({pct:.1f}%).")
    if unresolved:
        print(f"  Left {unresolved:,} rows unlocated: their block number exists "
              f"in another town and no candidate matched this one.")
    print(f"Saved enriched property info to: {PROPERTY_CSV}\n")
    return PROPERTY_CSV


if __name__ == "__main__":
    enrich_property_info()
