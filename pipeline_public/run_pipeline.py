#!/usr/bin/env python3
"""
run_pipeline.py
Master orchestration script to revive, update, and train the HDB Price Predictor pipeline.
Steps:
  1. Fetch latest resale data + HDB Resale Price Index from Data.gov.sg
  2. Build / Update pipeline_public/data/hdb_postal_data.json
  3. Feature engineering, scaling, and XGBoost training (hdb_model.ubj, hdb_model.json & hdb_scaler.joblib)
"""

import argparse
import os
import subprocess
import sys
import time
from importlib import import_module

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Imported after PROJECT_ROOT joins sys.path, since that is where
# pipeline_common lives.
from pipeline_common import find_compatible_python

# Silence optional C-extension warnings from older Anaconda packages
os.environ["PANDAS_USE_BOTTLENECK"] = "0"

def main():
    parser = argparse.ArgumentParser(description="HDB Price Predictor Ingestion & Training Pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching latest data from Data.gov.sg")
    parser.add_argument("--force-fetch", action="store_true", help="Bypass cache and force re-download of all records from Data.gov.sg")
    parser.add_argument("--skip-postal", action="store_true", help="Skip building postal_data.json and the MRT proximity lookup")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-map", action="store_true", help="Skip building interactive HDB resale map")
    args = parser.parse_args()

    start_total = time.time()
    print("================================================================")
    print("    HDB RESALE PRICE PREDICTOR - DATA & TRAINING PIPELINE       ")
    print("================================================================\n")

    # Step 1: Fetch Data (Pure standard library: json, urllib, csv)
    if not args.skip_fetch:
        fetch_module = import_module("01_fetch_data")
        fetch_module.download_all_resale_data(force=args.force_fetch)
        # Official market index used to project prices past the data cutoff.
        fetch_module.download_property_info()
        fetch_module.download_geospatial_layers()
        fetch_module.download_price_index()
    else:
        print("[Step 1/4] Skipping data fetch (--skip-fetch specified).\n")

    # Step 2: Location and lookups (pure standard library).
    #
    # Order matters. The published property table carries no location, so it
    # is enriched with postal codes and coordinates first; the postal lookup
    # is built from that, and MRT distances are keyed off the result.
    if not args.skip_postal:
        enrich_module = import_module("02c_enrich_property_info")
        enrich_module.enrich_property_info()

        postal_module = import_module("02_build_postal_data")
        postal_module.build_postal_data()

        mrt_module = import_module("02b_build_mrt_proximity")
        mrt_module.build_mrt_proximity()
    else:
        print("[Step 2/4] Skipping postal data generation (--skip-postal specified).\n")

    # Step 3: Train Model (Requires xgboost & scikit-learn)
    if not args.skip_train:
        print("[Step 3/4] Preparing model training...")
        has_xgboost = False
        try:
            import xgboost
            has_xgboost = True
        except ImportError:
            has_xgboost = False

        if has_xgboost:
            train_module = import_module("03_train_model")
            train_module.train_and_export()
        else:
            # Current python environment does not have xgboost; check for compatible venv
            comp_python = find_compatible_python(PROJECT_ROOT)
            if comp_python:
                print(f"Current Python ({sys.executable}) lacks XGBoost.")
                print(f"Auto-delegating training to compatible environment: {comp_python}")
                train_script = os.path.join(os.path.dirname(__file__), "03_train_model.py")
                res = subprocess.run([comp_python, train_script])
                if res.returncode != 0:
                    print(f"Error: Training failed with returncode {res.returncode}")
            else:
                print("\n[!] Error: 'xgboost' is not installed in the active environment.")
                print("Please install requirements or run with a virtual environment:")
                print("  pip install -r ../requirements.txt")
    else:
        print("[Step 3/4] Skipping model training (--skip-train specified).\n")

    # Step 4: Build Interactive Resale Map (Node.js)
    if not args.skip_map:
        print("[Step 4/4] Generating Interactive HDB Resale Price Map...")
        map_script = os.path.join(os.path.dirname(__file__), "04_build_map.js")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.exists(map_script):
            try:
                res = subprocess.run(["node", map_script], cwd=project_root)
                if res.returncode != 0:
                    print(f"[!] Warning: Map generation exited with returncode {res.returncode}")
            except Exception as e:
                print(f"[!] Warning: Could not run Node.js script: {e}")
                print("    Please ensure Node.js is installed or run: node pipeline_public/04_build_map.js")
        else:
            print("[!] Notice: 04_build_map.js not found, skipping map generation.")
    else:
        print("[Step 4/4] Skipping map generation (--skip-map specified).\n")

    total_time = time.time() - start_total
    print("================================================================")
    print(f" HDB PIPELINE COMPLETE in {total_time:.1f}s")
    print(" Artifacts ready: model, scaler, postal_data, and clickable map!")
    print("================================================================")

if __name__ == "__main__":
    main()
