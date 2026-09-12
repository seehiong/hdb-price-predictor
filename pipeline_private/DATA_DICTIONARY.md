# 📖 Singapore Property Transactions Data Dictionary

This document details the schema, column definitions, data types, and semantic descriptions for the property datasets used in the private housing pipeline.

---

## 1. Extracted Dataset: `private_resi_transactions.csv`

Location: [`pipeline_private/data/private_resi_transactions.csv`](file:///f:/github-proj/hdb-price-predictor/pipeline_private/data/private_resi_transactions.csv)  
Source: Filtered subset of URA Resale transactions (83,848 rows, 10.09 MB).

The model trains on the **private (non-landed)** subset of this file — condominium, apartment and executive condominium, 71,538 rows. The remaining 12,305 landed transactions (terrace, semi-detached, detached and their strata variants) are present in the CSV but filtered out before training; see [docs/methodology.md](../docs/methodology.md) for the measurements behind that decision.

| Column | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `project` | `string` | `PARC ROSEWOOD` | Official URA development or project name. |
| `street` | `string` | `ROSEWOOD DRIVE` | Street or avenue address. |
| `market_segment` | `string` | `OCR` | URA planning region market segment: `CCR` (Core Central Region), `RCR` (Rest of Central Region), `OCR` (Outside Central Region). |
| `district` | `string` | `25` | Two-digit Singapore postal district code (`01` to `28`). |
| `area` | `float` | `48.0` | Net strata floor area in square metres (sqm). |
| `floor_range` | `string` | `01-05` | Storey / floor level band in 5-floor increments (e.g. `01-05`, `06-10`, `11-15`, `21-25`). |
| `no_of_units` | `int` | `1` | Number of residential units transacted (always 1 for clean individual sales). |
| `contract_date` | `date` | `2021-06-01` | Transaction date (ISO format `YYYY-MM-DD` or `YYYY-MM`). |
| `type_of_sale` | `string` | `Resale` | Nature of transaction: filtered for `Resale` (secondary market sales). |
| `price` | `float` | `545000.0` | Total transacted price in Singapore Dollars (SGD). |
| `price_per_sqft` | `float` | `1054.84` | Unit price per square foot (calculated as `price / (area * 10.7639)`). |
| `property_type` | `string` | `Condominium` | Property category: `Condominium`, `Apartment`, `Executive Condominium`, `Terrace`, `Semi-detached`, `Detached`, etc. |
| `tenure` | `string` | `99 yrs lease commencing from 2011` | Legal tenure description: `Freehold`, `999 yrs...`, or `99 yrs lease commencing from <YYYY>`. |

---

## 2. Upstream Dataset: `sg_property_transactions_basic*.csv`

Location: `pipeline_private/data/` — not committed; download from Kaggle if you want to regenerate section 1 from source.  
Source: [Singapore Property Transactions Dataset (Ryeo, Kaggle, CC BY 4.0)](https://www.kaggle.com/datasets/y2h1987/singapore-property-transactions-hdb-private/data).

This describes the **basic** distribution — 24 columns, 370,821 rows spanning 2017–2026 — which is what `sg_property_transactions_basic*.csv` contains. The dataset is also published in an *enriched* form carrying geocoded coordinates and proximity features (MRT, schools, hawker centres, parks, hospitals, CBD distance); those columns are **not** present in the files here and were removed from this table, which previously listed 17 of them.

Note that this pipeline does not read the Kaggle file directly for training. `01b_extract_kaggle_data.py` filters it down to URA resale rows and writes `private_resi_transactions.csv` (section 1), which is what the model and map consume.

| Column | Type | Description |
| :--- | :--- | :--- |
| `tx_id` | `string` | Synthetic unique row identifier. |
| `source` | `string` | Data origin: `HDB` (public housing) or `URA` (private residential). |
| `contract_date` | `date` | Transaction date (`YYYY-MM-DD`). |
| `year` | `int` | Year of transaction (`2017`–`2026`). |
| `quarter` | `string` | Quarter label (e.g. `2024Q3`). |
| `price_sgd` | `float` | Transaction price in Singapore Dollars. |
| `area_sqm` | `float` | Floor area in square metres. |
| `price_per_sqm` | `float` | Price per square metre (`price_sgd / area_sqm`). |
| `price_per_sqft` | `float` | Price per square foot (`price_sgd / (area_sqm * 10.7639)`). |
| `project_or_block` | `string` | URA project name or HDB block number. |
| `street` | `string` | Street name. |
| `town_or_district` | `string` | HDB town (e.g. `ANG MO KIO`) or URA postal district code (`01`–`28`). |
| `planning_region` | `string` | URA market segment (`CCR`/`RCR`/`OCR`). Empty for HDB. |
| `property_type` | `string` | HDB flat type (`3 ROOM`, `4 ROOM`, etc.) or URA property type (`Condominium`, `Apartment`, `Terrace`, etc.). |
| `property_subtype` | `string` | HDB flat model (`Improved`, `Maisonette`, etc.) or URA strata type (`Strata`/`Land`). |
| `storey_range` | `string` | Floor band (HDB e.g. `'07 TO 09'`; URA e.g. `'06-10'`). |
| `tenure` | `string` | Legal tenure description (`Freehold`, `'99 yrs lease commencing from 2021'`, or `'HDB 99-year leasehold'`). |
| `sale_type` | `string` | URA: `New Sale`, `Resale`, `Sub Sale`. HDB: always `Resale`. |
| `no_of_units` | `int` | Number of units in this transaction (URA only; HDB always 1). |
| `is_bulk_transaction` | `bool` | `True` when `no_of_units > 1` (en-bloc or multi-unit purchase). |
| `effective_price_per_unit` | `float` | Unit price (`price_sgd / no_of_units`). |
| `remaining_lease_years` | `float` | Numeric remaining lease in years. |
| `property_age_years` | `float` | Years elapsed since lease commencement / completion. |
| `data_quality_flag` | `string` | Data hygiene classification (`clean`, `landed_oversized`, `en_bloc_or_bulk`, `no_geocode`). |
