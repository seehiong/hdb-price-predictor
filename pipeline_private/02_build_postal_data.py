#!/usr/bin/env python3
"""
02_build_postal_data.py
Extracts private housing and condominium metadata (6-digit postal code, project name,
dwelling units, property type, and coordinates) from URANoofDwellingUnits.geojson.

Incorporates standard Singapore road name canonical abbreviations:
- Normalizes full road names and abbreviations ('AVENUE' <-> 'AVE', 'ROAD' <-> 'RD', etc.)
- Supports multi-tier resolution: (project, abbreviated_road_name), canonical project, and (x, y) coordinates.

Pure Python standard library (re, json, csv, html).
"""

import csv
import json
import os
import re
from html.parser import HTMLParser

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
DATA_DIR = os.path.join(PIPELINE_DIR, "data")

GEOJSON_FILE = os.path.join(DATA_DIR, "URANoofDwellingUnits.geojson")
TRANSACTIONS_CSV = os.path.join(DATA_DIR, "private_resi_transactions.csv")

OUTPUT_POSTAL_JSON = os.path.join(DATA_DIR, "private_postal_data.json")
OUTPUT_NONLANDED_JSON = os.path.join(DATA_DIR, "private_nonlanded_postal_data.json")
OUTPUT_PROJECT_MAP_JSON = os.path.join(DATA_DIR, "project_to_postal_map.json")

# Standard Singapore Road Name Abbreviations
STREET_ABBREVIATIONS = {
    "AVENUE": "AVE",
    "BUKIT": "BT",
    "CENTRAL": "CTRL",
    "CLOSE": "CL",
    "COMMONWEALTH": "C'WEALTH",
    "CRESCENT": "CRES",
    "DRIVE": "DR",
    "GARDEN": "GDN",
    "GARDENS": "GDNS",
    "HEIGHTS": "HTS",
    "JALAN": "JLN",
    "KAMPONG": "KG",
    "LORONG": "LOR",
    "MARKET": "MKT",
    "NORTH": "NTH",
    "PARK": "PK",
    "PLACE": "PL",
    "ROAD": "RD",
    "SAINT": "ST",
    "SOUTH": "STH",
    "SQUARE": "SQ",
    "STREET": "ST",
    "TANJONG": "TG",
    "TERRACE": "TER",
    "UPPER": "UPP",
}


def abbreviate_road_name(road_name: str) -> str:
    """Normalizes and applies standard Singapore road abbreviations."""
    if not road_name:
        return ""
    name = " " + re.sub(r"\s+", " ", str(road_name).upper().strip()) + " "
    for full, abbrev in STREET_ABBREVIATIONS.items():
        # Match word boundaries to prevent substring collisions (e.g. 'PARKWAY' vs 'PARK')
        pattern = r"\b" + re.escape(full) + r"\b"
        name = re.sub(pattern, abbrev, name)
    return name.strip()


def canonicalize_project_name(proj_name: str) -> str:
    """Cleans project name by removing common legal suffixes and punctuation."""
    if not proj_name:
        return ""
    name = str(proj_name).upper().strip()
    name = re.sub(r"^(THE|A)\s+", "", name)
    name = re.sub(r"\b(CONDOMINIUM|CONDO|APARTMENTS?|RESIDENCES?)\b", "", name)
    name = re.sub(r"[^A-Z0-9]", "", name)  # Strip punctuation like D'LEEDON -> DLEEDON
    return name.strip()


class AttributeTableParser(HTMLParser):
    """Parses key-value attribute tables formatted inside HTML Description tags."""
    def __init__(self):
        super().__init__()
        self.in_th = False
        self.in_td = False
        self.current_key = None
        self.data = {}

    def handle_starttag(self, tag, attrs):
        if tag == "th":
            self.in_th = True
        elif tag == "td":
            self.in_td = True

    def handle_endtag(self, tag):
        if tag == "th":
            self.in_th = False
        elif tag == "td":
            self.in_td = False

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self.in_th:
            self.current_key = text.upper()
        elif self.in_td and self.current_key:
            self.data[self.current_key] = text
            self.current_key = None


def feature_attributes(feature: dict) -> dict:
    """
    Returns a feature's attributes regardless of how the export encodes them.

    Data.gov.sg serves this layer two ways depending on vintage: older KML
    conversions bury the fields in an HTML table inside `Description`, while
    current GeoJSON downloads expose them as plain properties. Reading only the
    HTML form silently matched nothing against a freshly downloaded file and
    wrote out an empty lookup, so both are handled.
    """
    props = feature.get("properties", {}) or {}
    description = props.get("Description", "")
    if description:
        parsed = parse_description_attributes(description)
        if parsed:
            return parsed
    return {k: ("" if v is None else str(v)) for k, v in props.items()}


def parse_description_attributes(desc_html: str) -> dict:
    """Extracts dictionary of attributes from HTML description table."""
    if not desc_html:
        return {}
    parser = AttributeTableParser()
    try:
        parser.feed(desc_html)
        return parser.data
    except Exception:
        matches = re.findall(r"<th>([^<]+)</th>\s*<td>([^<]*)</td>", desc_html, re.IGNORECASE)
        return {k.strip().upper(): v.strip() for k, v in matches}


def build_private_postal_data():
    print("=== Step 2: Building Private Housing Postal & Address Lookup ===")
    os.makedirs(DATA_DIR, exist_ok=True)

    geojson_path = GEOJSON_FILE
    if not os.path.exists(geojson_path):
        print(f"[!] Error: Dwelling units file not found at {GEOJSON_FILE}")
        print("Please place 'URANoofDwellingUnits.geojson' in pipeline_private/data/ to proceed.")
        return None

    print(f"Reading URA Dwelling Units from: {geojson_path}")
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    print(f"Total features in GeoJSON: {len(features):,}")

    postal_lookup = {}
    project_map = {}
    canonical_project_map = {}

    for feat in features:
        attrs = feature_attributes(feat)

        postal = str(attrs.get("POSTALCODE", "")).strip().zfill(6)
        raw_proj = attrs.get("PROJ_NAME", "").strip().upper()
        prop_type = attrs.get("PROP_TYPE", "").strip()
        du = attrs.get("DU", "")
        x_addr = attrs.get("X_ADDR", "")
        y_addr = attrs.get("Y_ADDR", "")

        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        lon = coords[0] if len(coords) >= 2 else None
        lat = coords[1] if len(coords) >= 2 else None

        if not postal or postal == "000000" or not postal.isdigit():
            continue

        entry = {
            "postal_code": postal,
            "project_name": raw_proj,
            "property_type": prop_type,
            "dwelling_units": int(du) if du.isdigit() else None,
            "x": float(x_addr) if x_addr else None,
            "y": float(y_addr) if y_addr else None,
            "longitude": lon,
            "latitude": lat
        }

        # 1. Direct Postal Lookup
        if postal not in postal_lookup:
            postal_lookup[postal] = entry

        # 2. Raw Project Lookup
        if raw_proj:
            if raw_proj not in project_map:
                project_map[raw_proj] = []
            project_map[raw_proj].append(entry)

        # 3. Canonical Project Lookup (Stripped of suffixes, punctuation)
        canon_proj = canonicalize_project_name(raw_proj)
        if canon_proj:
            if canon_proj not in canonical_project_map:
                canonical_project_map[canon_proj] = []
            canonical_project_map[canon_proj].append(entry)

    print(f"Indexed {len(postal_lookup):,} postal codes and {len(project_map):,} unique projects.")

    # 4. Tally transactions using road abbreviation and canonical project resolution
    if os.path.exists(TRANSACTIONS_CSV):
        print(f"\nTallying with transaction records: {TRANSACTIONS_CSV}")
        matched_exact = 0
        matched_canon = 0
        unmatched = 0
        unmatched_samples = set()

        with open(TRANSACTIONS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                proj = row.get("project", "").strip().upper()
                street = abbreviate_road_name(row.get("street", ""))

                if proj in project_map:
                    matched_exact += 1
                else:
                    canon = canonicalize_project_name(proj)
                    if canon in canonical_project_map:
                        matched_canon += 1
                    else:
                        unmatched += 1
                        if proj:
                            unmatched_samples.add(f"{proj} ({street})")

        total = matched_exact + matched_canon + unmatched
        if total > 0:
            rate = ((matched_exact + matched_canon) / total) * 100
            print(f"  Exact Project Match:     {matched_exact:,}")
            print(f"  Canonical Project Match: {matched_canon:,}")
            print(f"  Overall Match Rate:      {matched_exact + matched_canon:,}/{total:,} ({rate:.1f}%)")
            if unmatched_samples:
                print(f"  Sample Unmatched: {list(unmatched_samples)[:5]}")

    # 5. Save outputs
    with open(OUTPUT_POSTAL_JSON, "w", encoding="utf-8") as f:
        json.dump(postal_lookup, f, indent=2)
    print(f"\n[OK] Saved master private postal lookup ({len(postal_lookup):,} entries) to: {OUTPUT_POSTAL_JSON}")

    # Also export compact condo/apartment subset for web app
    nonlanded_lookup = {
        k: v for k, v in postal_lookup.items()
        if any(t in str(v.get("property_type", "")).upper() for t in ["NON-LANDED", "CONDO", "APT", "APARTMENT", "EC", "EXECUTIVE CONDOMINIUM"])
    }
    with open(OUTPUT_NONLANDED_JSON, "w", encoding="utf-8") as f:
        json.dump(nonlanded_lookup, f, indent=2)
    print(f"[OK] Saved non-landed postal lookup ({len(nonlanded_lookup):,} entries) to: {OUTPUT_NONLANDED_JSON}")

    with open(OUTPUT_PROJECT_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(project_map, f, indent=2)
    print(f"[OK] Saved project-to-postal map to: {OUTPUT_PROJECT_MAP_JSON}\n")

    return OUTPUT_NONLANDED_JSON


if __name__ == "__main__":
    build_private_postal_data()
