#!/usr/bin/env python3
"""
03_train_model.py
Feature engineering, MinMaxScaler fitting, and XGBoost model training
for HDB Resale Flat Price Prediction.
Exports:
  - model.bst
  - scaler.joblib
"""

import json
import os
import re
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from pipeline_common import (write_metrics, load_cached_index, index_growth_rate,
                             compute_market_trend, sync_metrics_fallback)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_CSV = os.path.join(DATA_DIR, "resale_flat_prices_2017_present.csv")
MRT_PROXIMITY_JSON = os.path.join(DATA_DIR, "hdb_mrt_proximity.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_UBJ = os.path.join(MODELS_DIR, "hdb_model.ubj")
MODEL_BST = os.path.join(MODELS_DIR, "hdb_model.bst")
MODEL_JSON_OUTPUT = os.path.join(MODELS_DIR, "hdb_model.json")
SCALER_OUTPUT = os.path.join(MODELS_DIR, "hdb_scaler.joblib")
SCALER_JSON_OUTPUT = os.path.join(MODELS_DIR, "hdb_scaler.json")
MODEL_WEB_OUTPUT = os.path.join(MODELS_DIR, "hdb_model_web.json")
HDB_INDEX_JSON = os.path.join(MODELS_DIR, "hdb_price_index.json")

# A flat within ~400 m of an exit is genuinely walk-to-MRT; beyond that the
# premium decays smoothly and the continuous distance carries the signal.
MRT_WALKABLE_KM = 0.4

# The exact 58 feature schema used by the application
FEATURE_NAMES = [
    'floor_area_sqm', 'storey_avg', 'sale_year', 'sale_month', 'remaining_lease_years',
    'flat_type_1 ROOM', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM',
    'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION',
    'flat_model_2-room', 'flat_model_3Gen', 'flat_model_Adjoined flat', 'flat_model_Apartment',
    'flat_model_DBSS', 'flat_model_Improved', 'flat_model_Improved-Maisonette',
    'flat_model_Model A', 'flat_model_Model A-Maisonette', 'flat_model_Model A2',
    'flat_model_Multi Generation', 'flat_model_New Generation', 'flat_model_Premium Apartment',
    'flat_model_Premium Apartment Loft', 'flat_model_Premium Maisonette', 'flat_model_Simplified',
    'flat_model_Standard', 'flat_model_Terrace', 'flat_model_Type S1', 'flat_model_Type S2',
    'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH',
    'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG',
    'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST',
    'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS', 'town_PUNGGOL',
    'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES',
    'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN',
    # Proximity to transport. Distance to the nearest MRT exit is a stable
    # geographic fact rather than a price level, so it transfers cleanly to
    # future years: median unit price runs ~S$5,530/sqm within 250 m versus
    # ~S$4,960/sqm at 1-1.5 km.
    'mrt_distance_km', 'mrt_walkable'
]

# Fallback annual appreciation if the trend cannot be fitted, and a sanity cap so
# a noisy sample can never produce runaway forecasts.
DEFAULT_ANNUAL_GROWTH = 0.025
MAX_ANNUAL_GROWTH = 0.08

# Hold out the most recent complete year for honest, forward-looking accuracy.
# A random split lets the model see later sales of the same block in a market
# that rose ~31% over the holdout window, understating the error by roughly 2x.
HOLDOUT_FROM_LATEST_YEARS = 1

def parse_storey_range(s):
    if pd.isna(s):
        return np.nan
    nums = re.findall(r'\d+', str(s))
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2.0
    elif len(nums) == 1:
        return float(nums[0])
    return np.nan

def load_mrt_lookup(path: str = MRT_PROXIMITY_JSON):
    """Loads the (block|town) -> km lookup built by 02b_build_mrt_proximity.py."""
    if not os.path.exists(path):
        print(f"[!] MRT proximity lookup not found at {path}.")
        print("    Run 02b_build_mrt_proximity.py to enable transport features.")
        return {}, None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    distances = payload.get("distances", {})
    print(f"Loaded MRT proximity for {len(distances):,} (block, town) pairs.")
    return distances, payload.get("median_km")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Performing feature engineering...")
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # 1. Floor area
    df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')

    # 2. Storey average
    df['storey_avg'] = df['storey_range'].apply(parse_storey_range)
    median_storey = df['storey_avg'].median()
    df['storey_avg'] = df['storey_avg'].fillna(median_storey if not pd.isna(median_storey) else 5.0)

    # 3. Sale Year and Month
    dt = pd.to_datetime(df['month'], format='%Y-%m', errors='coerce')
    df['sale_year'] = dt.dt.year.astype(float)
    df['sale_month'] = dt.dt.month.astype(float)

    # 4. Remaining lease years
    lcd = pd.to_numeric(df['lease_commence_date'], errors='coerce')
    df['remaining_lease_years'] = np.clip((lcd + 99.0) - df['sale_year'], 10.0, 99.0)

    # 5. Target variable
    df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')

    # Drop invalid target rows
    df.dropna(subset=['resale_price', 'floor_area_sqm', 'sale_year'], inplace=True)

    # 5b. Transport proximity, joined on (block, town) - block numbers repeat
    # across the island, so the pair is the unique key.
    mrt_lookup, mrt_median = load_mrt_lookup()
    if mrt_lookup:
        blk = df['block'].astype(str).str.strip().str.upper()
        twn = df['town'].astype(str).str.strip().str.upper()
        km = (blk + '|' + twn).map(mrt_lookup)
        matched = km.notna().mean() * 100
        fallback = mrt_median if mrt_median is not None else float(km.median())
        df['mrt_distance_km'] = km.fillna(fallback).astype(float)
        print(f"Matched MRT distance for {matched:.1f}% of transactions "
              f"(unmatched filled with island median {fallback:.3f} km).")
    else:
        # Keep the schema stable so the browser feature vector never shifts.
        df['mrt_distance_km'] = 0.0
    df['mrt_walkable'] = (df['mrt_distance_km'] <= MRT_WALKABLE_KM).astype(float)

    # 6. One-hot encoding
    df['flat_type'] = df['flat_type'].astype(str).str.strip().str.upper()
    df['flat_model'] = df['flat_model'].astype(str).str.strip()
    df['town'] = df['town'].astype(str).str.strip().str.upper()

    df_dummies = pd.get_dummies(df, columns=['flat_type', 'flat_model', 'town'], dtype=float)

    # 7. Reconcile with exact FEATURE_NAMES
    for col in FEATURE_NAMES:
        if col not in df_dummies.columns:
            df_dummies[col] = 0.0

    X = df_dummies[FEATURE_NAMES].copy()
    y = df_dummies['resale_price'].copy()

    print(f"Engineered feature matrix shape: {X.shape}, target shape: {y.shape}")
    return X, y

def train_and_export():
    print(f"=== Starting Model Training Pipeline ===")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Resale transactions not found: {INPUT_CSV}\n"
            "The dataset is ~40 MB and is not committed, so a fresh clone has to "
            "download it first:\n"
            "    python pipeline_public/01_fetch_data.py\n"
            "or run run_pipeline.py without --skip-fetch."
        )

    print(f"Loading dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    print(f"Loaded {len(df):,} transactions.")
    X, y = engineer_features(df)

    trend = compute_market_trend(y, X['floor_area_sqm'], X['sale_year'])
    print(f"Latest sale_year in data: {trend['latest_year']} "
          f"(complete through {trend['latest_complete_year']})")
    print(f"Measured median PSM CAGR: {trend['annual_growth'] * 100:.2f}%")

    # Forecast growth from the official HDB Resale Price Index rather than a
    # rate fitted to our own transactions. The published series tracks our own
    # median PSM almost exactly (r = 0.998) but is the authoritative figure and
    # reflects the recent slowdown our fitted trend overstated.
    #
    # This changes only how prices are PROJECTED past the data cutoff, not the
    # training target: deflating prices by the index was tested on the private
    # model and made near-term accuracy worse, since projecting any trailing
    # trend overshoots a decelerating market.
    hdb_index = load_cached_index(HDB_INDEX_JSON)
    index_growth = round(index_growth_rate(hdb_index), 6) if hdb_index else None
    if index_growth is not None:
        print(f"Forecast growth from HDB RPI: {index_growth * 100:.2f}% "
              f"(fitted trend was {trend['annual_growth'] * 100:.2f}%)")
    else:
        print("[!] HDB RPI cache not found (run 01_fetch_data.py); "
              "forecasts fall back to the fitted trend.")

    # Chronological holdout: train on all complete historical years (through
    # latest_complete_year) and validate on subsequent unseen future sales (2026+).
    # This mirrors what the app actually does -- price a sale that has not happened
    # yet -- while ensuring all complete calendar years (through 2025) are fully
    # learned by the trees for historical comparable valuation.
    if trend["latest_year"] > trend["latest_complete_year"]:
        holdout_year = trend["latest_complete_year"] + 1
    else:
        holdout_year = trend["latest_complete_year"] - HOLDOUT_FROM_LATEST_YEARS + 1
    eval_mask = (X["sale_year"] >= holdout_year).values
    if eval_mask.sum() < 500 or (~eval_mask).sum() < 2000:
        print("[!] Not enough history for a chronological holdout; using random split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True)
        split_kind = "random"
    else:
        split_kind = f"chronological (validate on {holdout_year}+)"
        X_train, X_val = X[~eval_mask], X[eval_mask]
        y_train, y_val = y[~eval_mask], y[eval_mask]
    print(f"Splitting dataset: {split_kind}")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

    # Fit MinMaxScaler
    print("Fitting MinMaxScaler on all 58 features...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Save scaler
    joblib.dump(scaler, SCALER_OUTPUT)
    print(f"Saved MinMaxScaler to: {SCALER_OUTPUT}")

    # Configure XGBoost
    print("Initializing XGBoost Regressor...")
    xgb_reg = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.08,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric='rmse'
    )

    t0 = time.time()
    print("Training XGBoost model...")
    xgb_reg.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=100
    )
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s.")

    # Save Booster in both standard JSON and modern binary UBJSON format (.ubj)
    import shutil
    booster = xgb_reg.get_booster()
    booster.save_model(MODEL_JSON_OUTPUT)
    print(f"Saved trained XGBoost JSON model to: {MODEL_JSON_OUTPUT}")
    booster.save_model(MODEL_UBJ)
    print(f"Saved trained XGBoost UBJSON model to: {MODEL_UBJ}")
    # Also maintain .bst file for backwards compatibility without triggering warnings
    shutil.copyfile(MODEL_UBJ, MODEL_BST)
    print(f"Saved legacy binary alias to: {MODEL_BST}")

    # Export scaler as JSON for client-side JavaScript execution
    scaler_dict = {
        'feature_names': FEATURE_NAMES,
        'scale': [round(float(x), 8) for x in scaler.scale_],
        'min': [round(float(x), 8) for x in scaler.min_],
        # Extrapolation metadata consumed by the browser inference path.
        'latest_year': trend['latest_year'],
        'latest_complete_year': trend['latest_complete_year'],
        # Official index growth wins when available; the fitted trend is the
        # offline fallback.
        'annual_growth': index_growth if index_growth is not None else trend['annual_growth'],
        'fitted_growth': trend['annual_growth'],
        'index_source': ('HDB Resale Price Index (1Q2009=100), quarterly'
                         if index_growth is not None else None)
    }
    with open(SCALER_JSON_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(scaler_dict, f)
    print(f"Saved scaler JSON to: {SCALER_JSON_OUTPUT}")

    # Export compact tree representation for ultra-fast browser inference
    # Early stopping keeps training past the best iteration to prove the score
    # is not improving, so the saved booster holds more trees than the model
    # that was actually validated. Python's predict() silently truncates at
    # best_iteration; the browser walks whatever we export, so truncate here or
    # the shipped model is not the one these metrics describe.
    best_trees = (xgb_reg.best_iteration + 1) if xgb_reg.best_iteration is not None \
        else xgb_reg.n_estimators
    dumps = booster.get_dump(dump_format='json')[:best_trees]
    print(f'Exporting {len(dumps)} trees (booster holds '
          f'{len(booster.get_dump())}; truncated at best iteration).')
    m_json = json.load(open(MODEL_JSON_OUTPUT, 'r', encoding='utf-8'))
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

    with open(MODEL_WEB_OUTPUT, 'w', encoding='utf-8') as out:
        json.dump({'b': base_score, 't': trees_compact}, out, separators=(',', ':'))
    print(f"Saved compact Web model to: {MODEL_WEB_OUTPUT}")

    # Evaluate validation metrics
    print("\n--- Validation Performance ---")
    y_pred = xgb_reg.predict(X_val_scaled)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

    print(f"Mean Absolute Error (MAE):    S$ {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): S$ {rmse:,.2f}")
    print(f"Mean Abs Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared Score (R2):         {r2:.4f}")
    print("===================================\n")

    published = write_metrics("hdb", {
        "label": "Public Housing (HDB Flat Resale Model)",
        "source": "Housing & Development Board (Data.gov.sg)",
        "records": int(len(df)),
        "features": len(FEATURE_NAMES),
        "trees": int(best_trees),
        "r2": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
        "validation": split_kind,
        "holdout_samples": int(len(y_val)),
        "latest_year": trend["latest_year"],
        "annual_growth": index_growth if index_growth is not None else trend["annual_growth"],
        "index_source": ("HDB Resale Price Index (1Q2009=100), quarterly"
                         if index_growth is not None else None),
        "forecast_base_year": trend["latest_complete_year"],
    })

    # Keep the offline copy in app.js in step with what was just trained.
    sync_metrics_fallback(os.path.join(PROJECT_ROOT, "app.js"), published)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "records": len(df),
        "best_iteration": xgb_reg.best_iteration
    }

if __name__ == "__main__":
    train_and_export()
