#!/usr/bin/env python3
"""
run_pipeline.py
Master orchestration script for the Private Condo Resale Ingestion & Training Pipeline.
Steps:
  1. Fetch private transactions from URA Data Service API (or use existing CSV)
  2. Build / Update pipeline_private/data/private_nonlanded_postal_data.json and project-to-postal mapping
  3. Feature engineering, scaling, and XGBoost training (condo_model.ubj, condo_model.json & condo_scaler.joblib)
"""

import argparse
import os
import subprocess
import sys
import time
from importlib import import_module

# Silence optional C-extension warnings
os.environ["PANDAS_USE_BOTTLENECK"] = "0"
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)

# Add current pipeline directory to sys.path so modules can be imported
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imported after PROJECT_ROOT joins sys.path, since that is where
# pipeline_common lives.
from pipeline_common import find_compatible_python


def main():
    parser = argparse.ArgumentParser(description="Private Condo Resale Ingestion & Training Pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching latest data from URA")
    parser.add_argument("--skip-postal", action="store_true", help="Skip building private_postal_data.json")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-map", action="store_true", help="Skip building interactive condo resale map")
    args = parser.parse_args()

    start_total = time.time()
    print("================================================================")
    print("   PRIVATE CONDO RESALE PREDICTOR - DATA & TRAINING PIPELINE    ")
    print("================================================================\n")

    # Step 1: Fetch Data (Pure standard library)
    if not args.skip_fetch:
        fetch_module = import_module("01_fetch_data")
        fetch_module.download_private_data()

        # Official market index used to project prices past the data cutoff.

        fetch_module.download_price_index()
    else:
        print("[Step 1/4] Skipping data fetch (--skip-fetch specified).\n")

    # Step 2: Build Postal Lookup (Pure standard library)
    if not args.skip_postal:
        postal_module = import_module("02_build_postal_data")
        postal_module.build_private_postal_data()
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
            comp_python = find_compatible_python(PROJECT_ROOT)
            if comp_python:
                print(f"Current Python ({sys.executable}) lacks XGBoost.")
                print(f"Auto-delegating training to compatible environment: {comp_python}")
                train_script = os.path.join(PIPELINE_DIR, "03_train_model.py")
                res = subprocess.run([comp_python, train_script])
                if res.returncode != 0:
                    print(f"Error: Training failed with returncode {res.returncode}")
            else:
                print("\n[!] Error: 'xgboost' is not installed in the active environment.")
                print("Please install requirements or run with uv:")
                print("  uv run python pipeline_private/run_pipeline.py")
    else:
        print("[Step 3/4] Skipping model training (--skip-train specified).\n")

    # Step 4: Build Interactive Resale Map (Node.js)
    if not args.skip_map:
        print("[Step 4/4] Generating Interactive Private Condo Resale Map...")
        map_script = os.path.join(PIPELINE_DIR, "04_build_map.js")
        if os.path.exists(map_script):
            try:
                res = subprocess.run(["node", map_script], cwd=PROJECT_ROOT)
                if res.returncode != 0:
                    print(f"[!] Warning: Map generation exited with returncode {res.returncode}")
            except Exception as e:
                print(f"[!] Warning: Could not run Node.js script: {e}")
                print("    Please ensure Node.js is installed or run: node pipeline_private/04_build_map.js")
        else:
            print("[!] Notice: 04_build_map.js not found, skipping map generation.")
    else:
        print("[Step 4/4] Skipping map generation (--skip-map specified).\n")

    total_time = time.time() - start_total
    print("================================================================")
    print(f" PRIVATE PIPELINE COMPLETE in {total_time:.1f}s")
    print(" Artifacts ready: model, scaler, private_nonlanded_postal_data, and clickable map!")
    print("================================================================\n")


if __name__ == "__main__":
    main()
