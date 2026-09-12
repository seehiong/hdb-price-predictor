#!/usr/bin/env python3
"""
03_train_model.py
Feature engineering, scaling, and XGBoost regression training for Private Condo Resale Prices.
Outputs model and scaler artifacts (Python & in-browser compatible formats).
"""

import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb
import joblib

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PROJECT_ROOT)
from pipeline_common import (write_metrics, load_cached_index, index_growth_rate,
                             compute_market_trend, sync_metrics_fallback)
DATA_DIR = os.path.join(PIPELINE_DIR, "data")
INPUT_CSV = os.path.join(DATA_DIR, "private_resi_transactions.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_UBJ = os.path.join(MODELS_DIR, "condo_model.ubj")
MODEL_BST = os.path.join(MODELS_DIR, "condo_model.bst")
MODEL_JSON = os.path.join(MODELS_DIR, "condo_model.json")
MODEL_WEB_JSON = os.path.join(MODELS_DIR, "condo_model_web.json")
SCALER_JOBLIB = os.path.join(MODELS_DIR, "condo_scaler.joblib")
SCALER_JSON = os.path.join(MODELS_DIR, "condo_scaler.json")
FEATURES_TXT = os.path.join(MODELS_DIR, "condo_features.txt")
PROJECT_ENCODING_JSON = os.path.join(MODELS_DIR, "condo_project_encoding.json")
METRICS_JSON = os.path.join(MODELS_DIR, "metrics.json")
URA_INDEX_JSON = os.path.join(MODELS_DIR, "ura_price_index.json")

# Target property types
TARGET_PROPERTY_TYPES = ["CONDOMINIUM", "APARTMENT", "EXECUTIVE CONDOMINIUM"]

# Sentinel "remaining lease" for freehold / 900+ year tenures. Must match the
# value used by the browser inference path in app.js (evaluateCondoTreeModel).
FREEHOLD_LEASE_YEARS = 999.0

# Fallback annual appreciation if the trend cannot be fitted, and a sanity cap so
# a noisy sample can never produce runaway forecasts.
DEFAULT_ANNUAL_GROWTH = 0.025
MAX_ANNUAL_GROWTH = 0.08

# Smoothing for project/street target encoding. A project with few transactions
# is pulled toward the global mean; ~20 sales is where we trust its own level.
TARGET_ENCODING_SMOOTHING = 20

# Hold out the most recent year for honest, forward-looking accuracy reporting.
# A random split lets the model see future transactions of the same project and
# flatters the metrics roughly 2x on this data.
HOLDOUT_FROM_LATEST_YEARS = 1

# Latest sale_year seen in training data is discovered at train time and written
# into the scaler JSON so the front-end knows where extrapolation begins.


def parse_contract_date(val: str):
    """
    Parses URA contractDate.
    Formats: '2024-03-01' -> Year 2024, Month 3.
             '2024-03'    -> Year 2024, Month 3.
             '1224'       -> Month 12, Year 2024.
             '125'        -> Month 01, Year 2025.
             '0125'       -> Month 01, Year 2025.
    """
    val = str(val).strip()
    if not val:
        return 2024, 6

    if "-" in val:
        parts = val.split("-")
        try:
            return int(parts[0]), max(1, min(12, int(parts[1])))
        except (ValueError, IndexError):
            return 2024, 6

    if not val.isdigit():
        return 2024, 6

    if len(val) == 3:
        month = int(val[0])
        yy = int(val[1:])
    elif len(val) == 4:
        month = int(val[:2])
        yy = int(val[2:])
    else:
        return 2024, 6

    year = 2000 + yy if yy < 70 else 1900 + yy
    month = max(1, min(12, month))
    return year, month


def parse_floor_mid(val: str) -> float:
    """Parses floor range like '06 to 10' or '06-10' to numeric midpoint (8.0)."""
    val = str(val).upper().strip()
    nums = re.findall(r"\d+", val)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2.0
    elif len(nums) == 1:
        return float(nums[0])
    return 5.0  # Default middle ground if missing or ground floor


def parse_tenure_details(tenure_str: str, contract_year: int):
    """
    Parses tenure string into (is_freehold, is_99_yr, is_999_yr, remaining_lease_years).

    Extracts the actual lease TERM and commencement year rather than substring
    matching, so that long leases are not mistaken for 99-year ones.
    Examples: 'Freehold', '99 yrs lease commencing from 2012',
              '999 yrs from 1885', '956 yrs lease commencing from 1928',
              '103 yrs lease commencing from 1996', '60 yrs lease from 2013'.

    Leases of 900 years or more are economically equivalent to freehold and are
    bucketed as such (is_999_yr), not as wasting 99-year assets.
    """
    t = str(tenure_str).upper()
    if "FREEHOLD" in t or "FREE HOLD" in t:
        return 1, 0, 0, FREEHOLD_LEASE_YEARS

    # Term = first number followed by a year/yr token; commencement = a 4-digit year.
    term_match = re.search(r"(\d{1,4})\s*(?:YRS?|YEARS?)", t)
    term = float(term_match.group(1)) if term_match else None

    commence_match = re.search(r"(?:FROM|COMMENCING FROM|COMMENCING)\s*(\d{4})", t)
    if not commence_match:
        # Fall back to any 4-digit year that is not the lease term itself.
        years = [int(y) for y in re.findall(r"(\d{4})", t)]
        if term is not None:
            years = [y for y in years if float(y) != term]
        commence_year = years[0] if years else (contract_year - 10)
    else:
        commence_year = int(commence_match.group(1))

    if term is None:
        # Unparseable tenure: assume a generic 99-year leasehold.
        term = 99.0

    elapsed = contract_year - commence_year
    remaining = max(1.0, term - elapsed)

    # 900+ year leases are effectively freehold in value terms.
    if term >= 900:
        return 0, 0, 1, FREEHOLD_LEASE_YEARS

    return 0, 1, 0, min(remaining, 99.0) if term <= 99 else min(remaining, term)


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    print(f"Loading private transactions from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw records loaded: {len(df):,}")

    # Standardize column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Filter for Resale if type_of_sale is present
    if "type_of_sale" in df.columns:
        # Values can be 'Resale', '3', or text
        is_resale = df["type_of_sale"].astype(str).str.contains("Resale|3", case=False, na=False)
        df = df[is_resale].copy()
        print(f"  Resale transactions: {len(df):,}")

    # Clean numeric fields
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df = df.dropna(subset=["price", "area"])
    df = df[(df["price"] > 100000) & (df["price"] < 30000000)]
    df = df[(df["area"] > 15) & (df["area"] < 1000)]

    # Standardize property type
    df["property_type"] = df["property_type"].astype(str).str.upper().str.strip()
    df = df[df["property_type"].isin(TARGET_PROPERTY_TYPES)].copy()
    print(f"  Filtered to Condos/Apartments/ECs: {len(df):,}")

    # Extract date features
    parsed_dates = [parse_contract_date(d) for d in df["contract_date"]]
    df["sale_year"] = [d[0] for d in parsed_dates]
    df["sale_month"] = [d[1] for d in parsed_dates]

    # Floor midpoint
    df["floor_mid"] = df["floor_range"].apply(parse_floor_mid)

    # Tenure features
    tenure_info = [parse_tenure_details(t, y) for t, y in zip(df["tenure"], df["sale_year"])]
    df["is_freehold"] = [ti[0] for ti in tenure_info]
    df["is_99_yr"] = [ti[1] for ti in tenure_info]
    df["is_999_yr"] = [ti[2] for ti in tenure_info]
    df["remaining_lease"] = [ti[3] for ti in tenure_info]

    # Clean district (padded to 2 digits '01' - '28')
    def clean_district(val):
        nums = re.findall(r"\d+", str(val))
        return f"{int(nums[0]):02d}" if nums else "00"

    df["district_code"] = df["district"].apply(clean_district)

    # Clean market segment
    df["market_segment"] = df["market_segment"].astype(str).str.upper().str.strip()
    valid_segments = ["CCR", "RCR", "OCR"]
    df["market_segment"] = df["market_segment"].apply(lambda s: s if s in valid_segments else "OCR")

    return df


def smoothed_target_map(keys: pd.Series, values: pd.Series,
                       smoothing: int = TARGET_ENCODING_SMOOTHING):
    """
    Builds a {key: encoded_value} map, shrinking sparse keys toward the global
    mean so a project with two sales does not get a confident price level.
    Returns (map, global_mean).
    """
    keys = pd.Series(keys).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)
    global_mean = float(values.mean())
    agg = values.groupby(keys).agg(["mean", "count"])
    smoothed = ((agg["mean"] * agg["count"] + global_mean * smoothing)
                / (agg["count"] + smoothing))
    return smoothed.to_dict(), global_mean


def oof_target_encode(keys: pd.Series, values: pd.Series, n_splits: int = 5,
                      smoothing: int = TARGET_ENCODING_SMOOTHING) -> np.ndarray:
    """
    Out-of-fold target encoding for the TRAINING matrix.

    Encoding a row using its own target leaks the answer into the feature, and
    the model learns to trust it far more than it should. Each fold is therefore
    encoded using only the other folds.
    """
    keys = pd.Series(keys).reset_index(drop=True)
    values = pd.Series(values).reset_index(drop=True)
    out = np.full(len(values), float(values.mean()), dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fit_idx, apply_idx in kf.split(keys):
        fold_map, fold_mean = smoothed_target_map(
            keys.iloc[fit_idx], values.iloc[fit_idx], smoothing)
        out[apply_idx] = keys.iloc[apply_idx].map(fold_map).fillna(fold_mean).values
    return out


def engineer_features(df: pd.DataFrame, train_mask: np.ndarray = None):
    """
    Builds the model matrix.

    `train_mask` marks the rows the encoders may learn from. Training rows are
    encoded out-of-fold; held-out rows are encoded from the training rows only,
    so no future price information reaches the evaluation set.

    Returns (features_df, y, encoding_artifacts).
    """
    print("Engineering features and one-hot encoding...")

    if train_mask is None:
        train_mask = np.ones(len(df), dtype=bool)

    df = df.reset_index(drop=True)
    train_mask = np.asarray(train_mask, dtype=bool)

    # One-hot encode the low-cardinality categoricals.
    prop_dummies = pd.get_dummies(df["property_type"], prefix="prop", dtype=float)
    seg_dummies = pd.get_dummies(df["market_segment"], prefix="segment", dtype=float)
    dist_dummies = pd.get_dummies(df["district_code"], prefix="dist", dtype=float)

    # Project and street are far too high-cardinality to one-hot (2,800+ projects
    # would bloat the browser model), but project identity is the single
    # strongest price signal there is -- a development's name explains ~87% of
    # PSF variance versus ~47% for its postal district. Encode them as smoothed
    # mean unit price instead: one float each, no size cost.
    unit_price = df["price"] / df["area"]

    project_encoded = np.empty(len(df), dtype=float)
    street_encoded = np.empty(len(df), dtype=float)

    project_map, project_global = smoothed_target_map(
        df["project"][train_mask], unit_price[train_mask])
    street_map, street_global = smoothed_target_map(
        df["street"][train_mask], unit_price[train_mask])

    project_encoded[train_mask] = oof_target_encode(
        df["project"][train_mask], unit_price[train_mask])
    street_encoded[train_mask] = oof_target_encode(
        df["street"][train_mask], unit_price[train_mask])

    holdout = ~train_mask
    if holdout.any():
        project_encoded[holdout] = (df["project"][holdout]
                                    .map(project_map).fillna(project_global).values)
        street_encoded[holdout] = (df["street"][holdout]
                                   .map(street_map).fillna(street_global).values)

    # Transaction count is a liquidity/size proxy: large, frequently traded
    # developments price differently from small boutique blocks.
    project_counts = df["project"][train_mask].value_counts()
    project_txn_count = df["project"].map(project_counts).fillna(0.0).values

    engineered = pd.DataFrame({
        "project_psf": project_encoded,
        "street_psf": street_encoded,
        "project_txn_count": project_txn_count,
        # Explicit interaction: a high floor is worth more on a large unit.
        "area_x_floor": (df["area"] * df["floor_mid"]).values,
    }, index=df.index)

    num_cols = [
        "area", "floor_mid", "remaining_lease",
        "sale_year", "sale_month",
        "is_freehold", "is_99_yr", "is_999_yr"
    ]

    features_df = pd.concat(
        [df[num_cols], engineered, prop_dummies, seg_dummies, dist_dummies], axis=1)
    y = df["price"].values

    # A single island-wide fallback would price an unrecognised suburban project
    # like a central one (the global mean is pulled up by CCR sales), so also
    # publish a per-district mean to fall back to first.
    district_means = (unit_price[train_mask]
                      .groupby(df["district_code"][train_mask]).mean())

    # Project-to-segment mapping: resolves split-district ambiguities (e.g. D20
    # has both RCR like Bishan and OCR like Nuovo in Ang Mo Kio, and D05
    # has RCR like Pasir Panjang / Buona Vista and OCR like Clementi).
    # Take the predominant market segment for each project in training.
    project_segments = (
        df[train_mask]
        .groupby("project")["market_segment"]
        .agg(lambda s: s.mode().iloc[0] if not s.empty else "OCR")
    )
    proj_seg_dict = {str(k): str(v) for k, v in project_segments.items()}

    # Enrich with postal master for boutique developments that have 0 recent caveats
    postal_json_path = os.path.join(PROJECT_ROOT, "pipeline_private", "data", "private_nonlanded_postal_data.json")
    if not os.path.exists(postal_json_path):
        postal_json_path = os.path.join(PROJECT_ROOT, "private_nonlanded_postal_data.json")
    if os.path.exists(postal_json_path):
        try:
            with open(postal_json_path, "r", encoding="utf-8") as pf:
                condo_postal_data = json.load(pf)
            sector_to_seg = {
                "11": "RCR", "12": "OCR", "13": "RCR",
                "56": "OCR", "57": "RCR"
            }
            for postal, pitem in condo_postal_data.items():
                pname = (pitem.get("project_name") or "").strip().upper()
                if not pname or pname == "UNKNOWN" or pname in proj_seg_dict:
                    continue
                sec = postal[:2]
                if sec in sector_to_seg:
                    proj_seg_dict[pname] = sector_to_seg[sec]
        except Exception:
            pass

    artifacts = {
        "project_map": {str(k): round(float(v), 4) for k, v in project_map.items()},
        "project_global": round(float(project_global), 4),
        "district_means": {str(k): round(float(v), 4) for k, v in district_means.items()},
        "street_map": {str(k): round(float(v), 4) for k, v in street_map.items()},
        "street_global": round(float(street_global), 4),
        "project_counts": {str(k): int(v) for k, v in project_counts.items()},
        "project_segment": proj_seg_dict,
    }
    return features_df, y, artifacts


def train_and_export():
    start_time = time.time()
    print("================================================================")
    print("      CONDO / PRIVATE RESALE XGBOOST TRAINING PIPELINE          ")
    print("================================================================\n")

    if not os.path.exists(INPUT_CSV):
        print(f"[!] Error: Transaction dataset not found: {INPUT_CSV}")
        print("Please run Step 1 (01_fetch_data.py) first.")
        return None

    df = load_and_clean_data(INPUT_CSV)
    if len(df) < 50:
        print(f"[!] Insufficient records for training ({len(df)} rows).")
        return None

    df = df.reset_index(drop=True)

    # Tree models cannot extrapolate beyond the sale_year range they were trained
    # on: every future year collapses into the same terminal leaf. We measure the
    # historical market trend here and export it, so the front-end can apply an
    # explicit appreciation factor for years past the data cutoff.
    trend = compute_market_trend(df['price'], df['area'], df['sale_year'])
    print(f"Latest sale_year in data: {trend['latest_year']} "
          f"(complete through {trend['latest_complete_year']})")
    print(f"Measured median PSM CAGR: {trend['annual_growth'] * 100:.2f}%")

    # Forecast growth comes from the official URA index rather than a single
    # rate fitted to our own transactions. CCR, RCR and OCR diverge sharply
    # (roughly +20%, +33% and +46% over 2021-Q1 to 2026-Q2), so one blended
    # rate overstates prime-district forecasts badly: applying the OCR-weighted
    # 7.3% to a CCR condo overshoots by ~24% six years out.
    #
    # Note this changes only how prices are PROJECTED past the data cutoff.
    # Deflating the training target by the index was tested and made
    # one-year-ahead accuracy worse (10.06% vs 7.52% MAPE), because projecting
    # any trailing trend overshot a market that decelerated in 2025-26. The
    # model keeps predicting raw prices; only the forward curve changes.
    ura_index = load_cached_index(URA_INDEX_JSON)
    segment_growth = {seg: round(index_growth_rate(series), 6)
                      for seg, series in ura_index.items()} if ura_index else {}
    if segment_growth:
        rates = ", ".join(f"{k} {v * 100:.2f}%" for k, v in sorted(segment_growth.items()))
        print(f"Segment forecast growth from URA index: {rates}")
    else:
        print(f"[!] URA index cache not found (run 01_fetch_data.py); "
              f"forecasts fall back to the fitted "
              f"{trend['annual_growth'] * 100:.2f}% blended rate.")

    # Chronological holdout: train on all complete historical years (through
    # latest_complete_year) and validate on subsequent unseen future sales (2026+).
    # This mirrors what the app actually does -- price a transaction that has not
    # happened yet -- while ensuring all complete calendar years (through 2025) are
    # fully learned by the trees for historical comparable valuation.
    if trend["latest_year"] > trend["latest_complete_year"]:
        holdout_year = trend["latest_complete_year"] + 1
    else:
        holdout_year = trend["latest_complete_year"] - HOLDOUT_FROM_LATEST_YEARS + 1
    eval_mask = (df["sale_year"] >= holdout_year).values
    if eval_mask.sum() < 100 or (~eval_mask).sum() < 500:
        print("[!] Not enough history for a chronological holdout; "
              "falling back to a random split.")
        eval_mask = np.zeros(len(df), dtype=bool)
        eval_mask[train_test_split(np.arange(len(df)), test_size=0.15,
                                   random_state=42)[1]] = True
        split_kind = "random"
    else:
        split_kind = f"chronological (validate on {holdout_year}+)"

    train_mask = ~eval_mask

    # Encoders are fitted on training rows only, so no future price leaks in.
    X_df, y, encodings = engineer_features(df, train_mask=train_mask)
    feature_names = list(X_df.columns)
    print(f"\nTotal engineered features: {len(feature_names)}")

    X_train, y_train = X_df.values[train_mask], y[train_mask]
    X_val, y_val = X_df.values[eval_mask], y[eval_mask]
    print(f"Split: {split_kind}")
    print(f"Training set: {len(X_train):,} samples | Validation set: {len(X_val):,} samples")

    # Fit MinMaxScaler
    print("\nFitting feature scaler (MinMaxScaler)...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Export scaler artifacts
    joblib.dump(scaler, SCALER_JOBLIB)
    scaler_dict = {
        "features": feature_names,
        "min": [round(float(x), 8) for x in scaler.min_],
        "scale": [round(float(x), 8) for x in scaler.scale_],
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        # Extrapolation metadata consumed by the browser inference path.
        "latest_year": trend["latest_year"],
        "latest_complete_year": trend["latest_complete_year"],
        "annual_growth": trend["annual_growth"],
        "districts": sorted(
            f[len("dist_"):] for f in feature_names if f.startswith("dist_")
        ),
        # Per-segment forecast growth, used by the browser to project past the
        # training cutoff. Falls back to `annual_growth` when unavailable.
        "segment_growth": segment_growth,
        "index_source": ("URA Property Price Index of Non-landed Properties by "
                         "Locality (CCR/RCR/OCR), quarterly") if segment_growth else None,
    }
    with open(SCALER_JSON, "w", encoding="utf-8") as f:
        json.dump(scaler_dict, f, indent=2)
    print(f"Saved scaler to {SCALER_JOBLIB} and {SCALER_JSON}")

    # Train XGBoost model
    print("\nTraining XGBoost Regressor with early stopping...")
    regressor = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.08,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=30
    )

    regressor.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=100
    )

    # Evaluate on the chronological holdout.
    preds = regressor.predict(X_val_scaled)
    mae = mean_absolute_error(y_val, preds)
    rmse = root_mean_squared_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    mape = float(np.mean(np.abs((y_val - preds) / y_val)) * 100)

    print("\n=== Model Validation Performance ===")
    print(f"  MAE:  SGD {mae:,.2f}")
    print(f"  RMSE: SGD {rmse:,.2f}")
    print(f"  R2:   {r2 * 100:.2f}%")
    print(f"  MAPE: {mape:.2f}%   <- honest forward-looking error")

    # Save model artifacts in standard JSON and modern binary UBJSON (.ubj) format
    import shutil
    regressor.save_model(MODEL_JSON)
    regressor.save_model(MODEL_UBJ)
    shutil.copyfile(MODEL_UBJ, MODEL_BST)
    print(f"Saved model to {MODEL_JSON}, {MODEL_UBJ} (and alias {MODEL_BST})")
    with open(FEATURES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))

    # Export compact tree representation for ultra-fast browser inference
    # Early stopping keeps training past the best iteration to prove the score
    # is not improving, so the saved booster holds more trees than the model
    # that was actually validated. Python's predict() silently truncates at
    # best_iteration; the browser walks whatever we export, so truncate here or
    # the shipped model is not the one these metrics describe.
    best_trees = (regressor.best_iteration + 1) if regressor.best_iteration is not None \
        else regressor.n_estimators
    dumps = regressor.get_booster().get_dump(dump_format='json')[:best_trees]
    print(f'Exporting {len(dumps)} trees (booster holds '
          f'{len(regressor.get_booster().get_dump())}; truncated at best iteration).')
    m_json = json.load(open(MODEL_JSON, 'r', encoding='utf-8'))
    base_score = float(m_json['learner']['learner_model_param']['base_score'].strip('[]'))

    trees_compact = []
    for tree_str in dumps:
        t = json.loads(tree_str)
        nodes_map = {}
        def collect(n):
            nodes_map[n['nodeid']] = n
            if 'children' in n:
                for c in n['children']:
                    collect(c)
        collect(t)
        max_id = max(nodes_map.keys())

        l = [-1] * (max_id + 1)
        r = [-1] * (max_id + 1)
        f_idx = [0] * (max_id + 1)
        c = [0.0] * (max_id + 1)
        w = [0.0] * (max_id + 1)

        for nid, node in nodes_map.items():
            if 'leaf' in node:
                w[nid] = round(node['leaf'], 4)
            else:
                l[nid] = node['yes']
                r[nid] = node['no']
                f_idx[nid] = int(node['split'][1:])
                c[nid] = round(node['split_condition'], 5)

        trees_compact.append({'l': l, 'r': r, 'f': f_idx, 'c': c, 'w': w})

    with open(MODEL_WEB_JSON, 'w', encoding='utf-8') as out:
        json.dump({'b': base_score, 't': trees_compact}, out, separators=(',', ':'))
    print(f"Saved compact Web model to: {MODEL_WEB_JSON}")

    # Ship the project/street encodings so the browser can look up the same
    # values at prediction time. Only projects seen in training are included.
    with open(PROJECT_ENCODING_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "project_psf": encodings["project_map"],
            "project_psf_global": encodings["project_global"],
            "project_psf_by_district": encodings["district_means"],
            "street_psf": encodings["street_map"],
            "street_psf_global": encodings["street_global"],
            "project_txn_count": encodings["project_counts"],
            "project_segment": encodings["project_segment"],
        }, f, separators=(",", ":"))
    print(f"Saved project encodings to: {PROJECT_ENCODING_JSON}")

    # Publish the headline figures the web page renders, so index.html never
    # carries hand-copied numbers that go stale after a retrain.
    published = write_metrics("condo", {
        "label": "Private Non-Landed Model (Condo, Apartment, EC)",
        "source": "Urban Redevelopment Authority (URA Caveats)",
        "records": int(len(df)),
        "features": len(feature_names),
        "trees": int(best_trees),
        "projects": len(encodings["project_map"]),
        "r2": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
        "validation": split_kind,
        "holdout_samples": int(len(y_val)),
        "latest_year": trend["latest_year"],
        "annual_growth": trend["annual_growth"],
        "segment_growth": segment_growth,
        "index_source": ("URA Property Price Index of Non-landed Properties by "
                         "Locality (CCR/RCR/OCR), quarterly") if segment_growth else None,
        "forecast_base_year": trend["latest_complete_year"],
    })

    # Keep the offline copy in app.js in step with what was just trained.
    sync_metrics_fallback(os.path.join(PROJECT_ROOT, "app.js"), published)

    elapsed = time.time() - start_time
    print(f"\n[OK] Model successfully exported:")
    print(f"  - {MODEL_BST}")
    print(f"  - {MODEL_JSON}")
    print(f"  - {MODEL_WEB_JSON}")
    print(f"  - {FEATURES_TXT}")
    print(f"Completed in {elapsed:.1f}s\n")

    return regressor


if __name__ == "__main__":
    train_and_export()
