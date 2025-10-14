# src/utils/io_utils.py
import os
import json
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_experiment_results(df, out_csv):
    ensure_dir(os.path.dirname(out_csv))
    df.to_csv(out_csv, index=False)
