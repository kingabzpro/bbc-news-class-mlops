"""
Download the BBC articles dataset from Kaggle.
"""

import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

os.makedirs(RAW_DATA_DIR, exist_ok=True)


def download_bbc_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        "jacopoferretti/bbc-articles-dataset", path=str(RAW_DATA_DIR), unzip=True
    )
    print(f"Downloaded BBC dataset to {RAW_DATA_DIR}")

    # Move bbc-news-data.csv to RAW_DATA_DIR
    src_csv = RAW_DATA_DIR / "archive (2)" / "bbc-news-data.csv"
    dst_csv = RAW_DATA_DIR / "bbc-news-data.csv"
    if src_csv.exists():
        os.replace(src_csv, dst_csv)
        print(f"Moved {src_csv} to {dst_csv}")

    # Remove everything in RAW_DATA_DIR except bbc-news-data.csv
    for item in RAW_DATA_DIR.iterdir():
        if item.name != "bbc-news-data.csv":
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil

                shutil.rmtree(item)
    print(f"Cleaned up {RAW_DATA_DIR}, only bbc-news-data.csv remains.")


if __name__ == "__main__":
    download_bbc_dataset()
