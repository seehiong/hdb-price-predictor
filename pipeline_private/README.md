# Private (Non-Landed) Resale Pipeline

This directory provides a self-contained, end-to-end Python pipeline for **Singapore Private Residential & Condominium Resale Transactions**, mirroring the architecture of the public HDB pipeline (`pipeline_public/`).

---

## 📊 Data Sources & Attribution

1. **Transaction Dataset (`private_resi_transactions.csv`):**
   * **Primary Source:** **Urban Redevelopment Authority (URA) Real Estate Data Service / Caveats** (`PMI_Resi_Transaction`).
   * **Distribution & Curation:** [Singapore Property Transactions (HDB & Private)](https://www.kaggle.com/datasets/y2h1987/singapore-property-transactions-hdb-private/data) by **Ryeo** on Kaggle.
   * **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
   * **Scope:** 83,848 verified private residential resale caveat transactions (2021–2026) across all 28 Singapore postal districts and Core Central (CCR), Rest of Central (RCR), and Outside Central (OCR) market segments.
   * **Extraction:** Filtered for `source == 'URA'`, `sale_type == 'Resale'`, and non-bulk transactions (`is_bulk_transaction == False`) via `pipeline_private/01b_extract_kaggle_data.py`.
   * **Version Control:** Standardized to a 10.09 MB CSV tracked directly in Git (`pipeline_private/data/private_resi_transactions.csv`).
   * **Data Dictionary:** See [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) for column definitions of the transaction files. The public counterpart is [`pipeline_public/DATA_DICTIONARY.md`](../pipeline_public/DATA_DICTIONARY.md).

2. **Dwelling Units & Postal Resolution:**
   * **Source:** **URA Number of Dwelling Units** (`URANoofDwellingUnits.geojson`, Data.gov.sg dataset `d_be71daeab5930f96b90ad2857454d876`).
   * Maps 2,431 private non-landed developments — condominiums, apartments and ECs — to 6-digit postal codes, coordinates and dwelling unit counts. Landed property is not covered.

3. **BCA Strata Title & Constitution Dates (`bca_mcst_registry.json`):**
   * **Source:** **Building and Construction Authority (BCA)** via Data.gov.sg dataset [`d_f988c57e16e99ad3a649aa04572efd1c`](https://data.gov.sg/datasets/d_f988c57e16e99ad3a649aa04572efd1c/view) (*MCST Information*).
   * **Scope:** 3,682 registered strata schemes with official MCST plan numbers, constitution/establishment dates, and development locations.
   * **Usage:** Enriches interactive map popups with verified strata registration dates and numbers, while cleanly differentiating strata-titled developments from single-owner non-strata rental estates.

---

## 🛠️ Setup & Configuration

### 1. (Optional) Live URA API Access Key
To fetch new batches directly from URA's API:
1. Register for a free URA Data Service key at [Singapore Government Developer Portal](https://developer.tech.gov.sg/) or [URA API Portal](https://www.ura.gov.sg/maps/api/).
2. Create or update `.env` in your project root:
   ```env
   URA_ACCESS_KEY=your_ura_access_key_here
   ```

*Note: The tracked `private_resi_transactions.csv` is already included in the repository, so live URA API credentials are fully optional.*

---

## 🚀 Running the Pipeline

Run the end-to-end pipeline with standard Python or `uv`:

```bash
# Extract resale records from downloaded Kaggle CSV:
python pipeline_private/01b_extract_kaggle_data.py

# Rebuild private postal mapping:
python pipeline_private/02_build_postal_data.py

# Retrain condo XGBoost model:
python pipeline_private/03_train_model.py

# Rebuild interactive Leaflet condo map with real caveat popups:
node pipeline_private/04_build_map.js
```

### Fast Orchestrator Flags (`run_pipeline.py`)

* **Skip fetch** (rebuild postal data and train model from tracked CSV):
  ```bash
  python pipeline_private/run_pipeline.py --skip-fetch
  ```
* **Skip postal resolution** (only fetch and train):
  ```bash
  python pipeline_private/run_pipeline.py --skip-postal
  ```
* **Skip model training** (only refresh datasets and postal lookup):
  ```bash
  python pipeline_private/run_pipeline.py --skip-train
  ```

---

## 📁 Script Breakdown

| File | Description |
| :--- | :--- |
| **`01b_extract_kaggle_data.py`** | Extracts and filters real URA resale transactions from `sg_property_transactions_basic.csv` into version-controlled `private_resi_transactions.csv`. |
| **`01_fetch_data.py`** | Downloads URA Property Price Index (PPI), official BCA MCST registry from Data.gov.sg, and (optional) live URA private resale transaction batches. |
| **`02_build_postal_data.py`** | Parses dwelling unit GeoJSON to generate `private_nonlanded_postal_data.json` and `project_to_postal_map.json`. Reports project match rate against transaction records. |
| **`03_train_model.py`** | Feature engineering (tenure, remaining lease, floor midpoint, district one-hot, market segment), fits MinMaxScaler, and trains an XGBoost Regressor. |
| **`01c_merge_historical_data.py`** | Merges a historical transaction export into `private_resi_transactions.csv`, deduplicating the overlapping period. URA's API only serves a rolling five-year window, so older transactions can only come from an export. |
| **`04_build_map.js`** | Compiles interactive standalone Leaflet map embedding up to 50 real URA resale caveat transactions per condo project (`MAX_TX_PER_PROJECT`), enriched with BCA MCST registration data. |
| **`run_pipeline.py`** | CLI orchestrator running the full ingestion and training workflow. |

---

## 📦 Output Artifacts

* `pipeline_private/data/private_resi_transactions.csv`: 83,848 real URA private resale records.
* `pipeline_private/data/private_nonlanded_postal_data.json`: Postal lookup for 2,431 non-landed developments (condominiums, apartments and ECs).
* `pipeline_private/data/bca_mcst_registry.json`: 3,682 BCA Management Corporation Strata Title records with constitution dates and postal code associations.
* `condo_resale_price_map_clickable.html`: Interactive Leaflet map with real caveat popup histories and BCA strata badges.
* `models/condo_model.json` & `models/condo_model_web.json`: Trained XGBoost condo pricing models.
* `models/condo_scaler.json`: Feature scaling configuration.
