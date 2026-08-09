"""
Step 0 of the pipeline: pull the ASVspoof2019 dataset from Kaggle.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings
-> API -> "Create New Token"). The `kaggle` pip package reads it automatically.

Usage:
    python -m pipeline.download_dataset
"""

import os
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "awsaf49/asvpoof-2019-dataset"
DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def main():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET} into {DATA_RAW_DIR} ...")
    api.dataset_download_files(DATASET, path=DATA_RAW_DIR, quiet=False)

    zip_path = os.path.join(DATA_RAW_DIR, "asvpoof-2019-dataset.zip")
    if os.path.exists(zip_path):
        print("Unzipping ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_RAW_DIR)
        print("Done. Raw data is under", DATA_RAW_DIR)
    else:
        print("Expected zip not found - check the extraction manually:", os.listdir(DATA_RAW_DIR))


if __name__ == "__main__":
    main()
