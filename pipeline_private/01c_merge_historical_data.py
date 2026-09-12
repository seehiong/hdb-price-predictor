#!/usr/bin/env python3
"""
01c_merge_historical_data.py
Merges a historical private-transaction dataset into private_resi_transactions.csv.

URA's PMI_Resi_Transaction API serves a rolling five-year window, so the live
feed cannot reach back past roughly 2021 and never will -- re-fetching next year
gains recent months and silently drops the oldest ones. A historical export is
the only way to hold older transactions.

This stage takes such an export, normalises it to the URA schema the rest of the
pipeline expects, and merges it with the existing file. The merge is additive:
rows already present are never replaced, so the authoritative URA API data always
wins over a third-party copy of the same transaction.

WHAT THIS IS FOR

Historical rows are added for the map and the transaction views -- being able to
see a project's price history back to 2018 rather than 2021. They are NOT
expected to improve model accuracy; that was measured, and the marginal value of
each additional year of history decays sharply:

    train from 2024   MAE S$175,237   8.03% MAPE
    train from 2023   MAE S$167,836   7.82%
    train from 2022   MAE S$162,470   7.64%
    train from 2021   MAE S$159,498   7.52%   (current)

Each older year buys roughly half of what the previous one did, so 2018-2020
would plausibly be worth about 1% of MAE. The trainer's own date filter decides
what it uses; see TRAIN_FROM_YEAR in 03_train_model.py if you want to exclude
older rows from training while keeping them for display.

DEDUPLICATION

The two sources overlap from 2021 onward, and the same transaction appears in
both with small formatting differences -- prices as "1,250,000" or 1250000.0,
areas rounded differently, project names cased differently. Matching on the raw
row would therefore keep both copies.

Rows are keyed on the fields that identify a transaction and cannot legitimately
differ between two records of it:

    project, street, contract month, area (rounded), floor range, price

Area is rounded to the nearest sqm and price to the nearest dollar before
hashing, so representation differences do not create phantom rows.
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PIPELINE_DIR, "data")
TARGET_CSV = os.path.join(DATA_DIR, "private_resi_transactions.csv")

# The schema the rest of the pipeline reads. Any historical source is mapped
# onto this before merging.
URA_COLUMNS = [
    "project", "street", "market_segment", "district", "area", "floor_range",
    "no_of_units", "contract_date", "type_of_sale", "price", "price_per_sqft",
    "property_type", "tenure",
]

# Column names seen in historical exports, mapped to the URA schema. A source
# only needs to supply enough of these to build the dedup key.
COLUMN_ALIASES = {
    "project": ["project", "project_name", "project_or_block", "development"],
    "street": ["street", "street_name", "road", "address"],
    "market_segment": ["market_segment", "segment", "planning_region", "region"],
    "district": ["district", "postal_district", "town_or_district"],
    "area": ["area", "area_sqm", "floor_area_sqm", "area_sqft"],
    "floor_range": ["floor_range", "storey_range", "floor_level", "storey"],
    "no_of_units": ["no_of_units", "units", "number_of_units"],
    "contract_date": ["contract_date", "sale_date", "transaction_date", "month", "date"],
    "type_of_sale": ["type_of_sale", "sale_type", "transaction_type"],
    "price": ["price", "price_sgd", "transacted_price", "resale_price"],
    "price_per_sqft": ["price_per_sqft", "psf", "unit_price_psf"],
    "property_type": ["property_type", "property_subtype", "type"],
    "tenure": ["tenure", "tenure_type", "lease"],
}

RESALE_TOKENS = ("resale", "3")

# Property types this pipeline models. A combined export may also carry HDB
# flats and landed housing; neither belongs in the private non-landed file.
PRIVATE_TYPES = {
    "CONDOMINIUM", "APARTMENT", "EXECUTIVE CONDOMINIUM", "EC",
}


def _norm_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def build_column_map(fieldnames) -> dict:
    """Maps a source file's headers onto the URA schema, best effort."""
    available = {_norm_header(f): f for f in fieldnames}
    mapping = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in available:
                mapping[target] = available[alias]
                break
    return mapping


def parse_month(value: str):
    """
    Reduces any contract-date form to 'YYYY-MM'.

    URA uses MMYY ('0125'), exports commonly use ISO dates or 'YYYY-MM'. The
    dedup key needs a month, not a day: the two sources disagree on exact dates
    for the same transaction often enough that day-level matching splits them.
    """
    v = str(value).strip()
    if not v:
        return ""
    iso = re.match(r"^(\d{4})[-/](\d{1,2})", v)
    if iso:
        return f"{iso.group(1)}-{int(iso.group(2)):02d}"
    if v.isdigit() and len(v) in (3, 4):
        month = int(v[0]) if len(v) == 3 else int(v[:2])
        yy = int(v[1:]) if len(v) == 3 else int(v[2:])
        year = 2000 + yy if yy < 70 else 1900 + yy
        return f"{year}-{max(1, min(12, month)):02d}"
    return v


def _num(value, default=None):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def transaction_key(row: dict):
    """
    Identity of a transaction, stable across formatting differences.

    Project and street are upper-cased and stripped of punctuation; area is
    rounded to the nearest sqm and price to the nearest dollar, because the two
    sources round them differently for the same sale.
    """
    def clean(v):
        return re.sub(r"[^A-Z0-9]+", " ", str(v).upper()).strip()

    area = _num(row.get("area"))
    price = _num(row.get("price"))
    return (
        clean(row.get("project", "")),
        clean(row.get("street", "")),
        parse_month(row.get("contract_date", "")),
        round(area) if area is not None else "",
        clean(row.get("floor_range", "")),
        round(price) if price is not None else "",
    )


def normalise_row(raw: dict, colmap: dict) -> dict:
    """Projects a source row onto the URA schema."""
    out = {c: "" for c in URA_COLUMNS}
    for target, source in colmap.items():
        out[target] = raw.get(source, "")

    out["contract_date"] = parse_month(out["contract_date"])

    # Some exports carry area in sqft; the pipeline expects the URA convention.
    if "area_sqft" in {_norm_header(v) for v in colmap.values()}:
        sqft = _num(out["area"])
        if sqft:
            out["area"] = round(sqft / 10.7639, 2)

    if not out.get("no_of_units"):
        out["no_of_units"] = "1"

    # Write numerics bare. Exports often carry thousands separators, and the
    # downstream loader uses pd.to_numeric, which rejects "1,068,639" and
    # silently drops the row -- the merge appears to succeed while the data
    # never reaches the model or the map.
    price, area = _num(out["price"]), _num(out["area"])
    if price is not None:
        out["price"] = f"{price:.0f}"
    if area is not None:
        out["area"] = f"{area:.2f}".rstrip("0").rstrip(".")

    if not out.get("price_per_sqft") and price and area:
        out["price_per_sqft"] = round(price / (area * 10.7639), 2)
    else:
        psf = _num(out.get("price_per_sqft"))
        if psf is not None:
            out["price_per_sqft"] = round(psf, 2)

    return out


def load_existing(path: str):
    if not os.path.exists(path):
        return [], set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, {transaction_key(r) for r in rows}


def merge_historical(source_csv: str, target_csv: str = TARGET_CSV,
                     resale_only: bool = True, dry_run: bool = False):
    if not os.path.exists(source_csv):
        print(f"[!] Source not found: {source_csv}")
        return None

    existing, seen = load_existing(target_csv)
    print(f"Existing transactions: {len(existing):,}")

    with open(source_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        colmap = build_column_map(reader.fieldnames or [])
        missing = [c for c in ("project", "contract_date", "price", "area")
                   if c not in colmap]
        if missing:
            print(f"[!] Source is missing fields needed to identify a transaction: "
                  f"{', '.join(missing)}")
            print(f"    Headers seen: {', '.join(reader.fieldnames or [])}")
            return None
        print(f"Mapped {len(colmap)}/{len(URA_COLUMNS)} columns from the source.")

        added, duplicates, skipped = [], 0, Counter()
        for raw in reader:
            row = normalise_row(raw, colmap)

            if resale_only:
                sale = str(row.get("type_of_sale", "")).strip().lower()
                if sale and not any(t in sale for t in RESALE_TOKENS):
                    skipped["not a resale"] += 1
                    continue

            # A combined export mixes HDB flats, landed housing and private
            # non-landed. Only the last belongs here -- merging HDB rows into
            # the private file would corrupt both the condo model and the map,
            # and the HDB pipeline already has that data from Data.gov.sg.
            ptype = str(row.get("property_type", "")).strip().upper()
            if ptype and ptype not in PRIVATE_TYPES:
                skipped[f"not private non-landed"] += 1
                continue

            if not row["contract_date"] or _num(row["price"]) is None:
                skipped["unusable row"] += 1
                continue

            key = transaction_key(row)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            added.append(row)

    print(f"  Already present (deduplicated): {duplicates:,}")
    for reason, n in skipped.items():
        print(f"  Skipped, {reason}: {n:,}")
    print(f"  New transactions: {len(added):,}")

    if added:
        years = Counter(r["contract_date"][:4] for r in added if r["contract_date"])
        print("  New rows by year: " +
              ", ".join(f"{y} {n:,}" for y, n in sorted(years.items())))

    if dry_run:
        print("\n[dry run] Nothing written.")
        return len(added)

    if not added:
        print("\nNothing to merge; target left unchanged.")
        return 0

    combined = existing + added
    combined.sort(key=lambda r: str(r.get("contract_date", "")), reverse=True)
    with open(target_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=URA_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)

    print(f"\nWrote {len(combined):,} transactions to: {target_csv}")
    return len(added)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a historical private-transaction export into "
                    "private_resi_transactions.csv, skipping duplicates.")
    parser.add_argument("source", help="CSV to merge in")
    parser.add_argument("--target", default=TARGET_CSV)
    parser.add_argument("--all-sale-types", action="store_true",
                        help="Keep new sales and sub-sales as well as resales")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be merged without writing")
    args = parser.parse_args()

    result = merge_historical(args.source, args.target,
                              resale_only=not args.all_sale_types,
                              dry_run=args.dry_run)
    sys.exit(0 if result is not None else 1)
