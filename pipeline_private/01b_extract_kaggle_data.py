#!/usr/bin/env python3
"""
01b_extract_kaggle_data.py
Extracts real URA private residential resale transactions from sg_property_transactions_basic.csv
(Dataset source: Singapore Property Transactions by Ryeo on Kaggle, CC BY 4.0, sourced from URA Data Service).

Filters for:
- source == 'URA'
- sale_type == 'Resale'
- is_bulk_transaction == 'False'
Outputs standardized private_resi_transactions.csv for model training and interactive map popups.
"""

import os
import csv

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PIPELINE_DIR, "data")
INPUT_CSV = os.path.join(DATA_DIR, "sg_property_transactions_basic.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "private_resi_transactions.csv")


def extract_ura_resale():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    print(f"Reading from {INPUT_CSV}...")

    fieldnames = [
        "project",
        "street",
        "market_segment",
        "district",
        "area",
        "floor_range",
        "no_of_units",
        "contract_date",
        "type_of_sale",
        "price",
        "price_per_sqft",
        "property_type",
        "tenure",
    ]

    count = 0
    skipped_bulk = 0
    with open(INPUT_CSV, "r", encoding="utf-8") as in_f:
        reader = csv.DictReader(in_f)
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                if row.get("source") != "URA":
                    continue
                if row.get("sale_type") != "Resale":
                    continue
                if row.get("is_bulk_transaction") == "True":
                    skipped_bulk += 1
                    continue

                writer.writerow(
                    {
                        "project": row.get("project_or_block", "").strip(),
                        "street": row.get("street", "").strip(),
                        "market_segment": row.get("planning_region", "").strip(),
                        "district": row.get("town_or_district", "").strip(),
                        "area": row.get("area_sqm", "").strip(),
                        "floor_range": row.get("storey_range", "").strip(),
                        "no_of_units": row.get("no_of_units", "1").strip(),
                        "contract_date": row.get("contract_date", "").strip(),
                        "type_of_sale": row.get("sale_type", "").strip(),
                        "price": row.get("price_sgd", "").strip(),
                        "price_per_sqft": row.get("price_per_sqft", "").strip(),
                        "property_type": row.get("property_type", "").strip(),
                        "tenure": row.get("tenure", "").strip(),
                    }
                )
                count += 1

    file_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"Successfully extracted {count:,} URA resale transactions (skipped {skipped_bulk} bulk deals).")
    print(f"Output saved to: {OUTPUT_CSV} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    extract_ura_resale()
