# HDB Price Predictor - Pipeline & Training Hub

This directory provides a lightweight, self-contained Python pipeline for data ingestion, address resolution, feature engineering, and model training.

## Recommended Setup with `uv`

Using [`uv`](https://docs.astral.sh/uv/) guarantees an isolated, reproducible virtual environment in seconds and prevents conflicts with global Anaconda or system Python packages:

```bash
# In project root: sync dependencies into a local .venv
uv sync
```

## Running the Pipeline

You can run the end-to-end pipeline directly with `uv run`:

```bash
# In project root:
uv run python pipeline_public/run_pipeline.py

# Or from inside the pipeline_public/ directory:
cd pipeline_public
uv run python run_pipeline.py
```

- **Smart Caching & Change Detection**:
  The pipeline automatically checks Data.gov.sg's total transaction count before downloading:
  - If the dataset is already identical to remote, download is **skipped completely in ~0.3s** (`[CACHE HIT]`).
  - If new transactions are published, it performs an **incremental fetch** from the last local offset, downloading only the newly added records and appending them to the CSV.
- **Force re-download** (bypass cache and download full historical dataset from scratch):
  ```bash
  uv run python pipeline_public/run_pipeline.py --force-fetch
  ```
- **Skip fetch** (re-train and rebuild postal lookup using already downloaded data):
  ```bash
  uv run python pipeline_public/run_pipeline.py --skip-fetch
  ```
- **Skip model training** (only refresh transactions and postal data):
  ```bash
  uv run python pipeline_public/run_pipeline.py --skip-train
  ```
- **Skip map building** (skip regenerating the interactive map):
  ```bash
  uv run python pipeline_public/run_pipeline.py --skip-map
  ```

## Pipeline Architecture

1. **`01_fetch_data.py`**: Fetches all resale flat records directly from Data.gov.sg API (`d_8b84c4ee58e3cfc0ece0d773c8ca6abc`) using Python's standard library into `pipeline_public/data/resale_flat_prices_2017_present.csv`.
2. **`02_build_postal_data.py`**: Cross-references `pipeline_public/data/hdb_property_info.csv` with transactions to produce/update `pipeline_public/data/hdb_postal_data.json`.
3. **`03_train_model.py`**: Engineers 58 standard features, fits `MinMaxScaler`, trains XGBoost with early stopping, and outputs both Python and in-browser artifacts (`hdb_model.json`, `hdb_model.ubj`, `hdb_scaler.joblib`, `hdb_scaler.json`, and `hdb_model_web.json`) to the project root.
**Data Dictionary:** [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) documents every column of the resale and property datasets, plus the derived postal and MRT lookups. The private counterpart is [`pipeline_private/DATA_DICTIONARY.md`](../pipeline_private/DATA_DICTIONARY.md).

4. **`04_build_map.js`**: Compiles coordinates, postal codes, building specs, and up to 50 recent resale transactions for all 9,741 HDB blocks across Singapore into a self-contained (~7.7 MB), responsive Leaflet map (`hdb_resale_price_map_clickable.html`). The cap is `MAX_TX_PER_BLOCK` at the top of the script; each transaction costs roughly 26 bytes of payload.
5. **`run_pipeline.py`**: Orchestrates the entire 4-stage flow and auto-detects compatible virtual environments if invoked with system Python.
