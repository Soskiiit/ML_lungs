import os
import pathlib
import zipfile

import kaggle


def download_dataset():
    kaggle.api.competition_download_files("ml-intensive-yandex-autumn-2023", path="dataset")
    dataset_path = pathlib.Path("dataset") / "data"

    if not os.path.exists(dataset_path):
        with zipfile.ZipFile("./dataset/ml-intensive-yandex-autumn-2023.zip", 'r') as zip_ref:
            zip_ref.extractall("./dataset/")


if __name__ == "__main__":
    download_dataset()
