# Welcome to Machine Learning Housing Corp.!
# Your task is to predict median house values in Californian districts,
# given a number of features from these districts.

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
print(housing.head())
print(housing.info())
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

# The following cell is not shown either in the book. It creates the images/end_to_end_project
# folder (if it doesn't already exist), and it defines the save_fig() function which is used
# through this notebook to save the figures in high-res for the book.
