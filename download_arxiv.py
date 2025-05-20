# download the ArXiv dataset using kagglehub and check that the json file is there

import kagglehub
import os
from datasets import load_dataset
import pandas as pd

def download_dataset():
    print("Downloading ArXiv dataset from Kaggle...")
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    print(f"Dataset downloaded to: {path}")

    print("\nFiles in the downloaded directory:")
    for root, dirs, files in os.walk(path):
        for file in files:
            name_file = os.path.join(root, file)
            print(name_file)

if __name__ == '__main__':
    download_dataset()