import os
import requests
import zipfile
import pandas as pd

# Dataset URL
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"

# Paths
DATA_DIR = os.path.join("data", "raw")
ZIP_PATH = os.path.join(DATA_DIR, "electricity.zip")
TXT_PATH = os.path.join(DATA_DIR, "household_power_consumption.txt")  # actual file inside ZIP
CSV_PATH = os.path.join(DATA_DIR, "electricity.csv")

def download_file():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading dataset...")
    response = requests.get(URL, stream=True)
    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)
    print(f"Downloaded: {ZIP_PATH}")

def unzip_file():
    print("Extracting ZIP...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print(f"Extracted TXT: {TXT_PATH}")

def convert_to_csv():
    print("Converting TXT (semicolon-separated) to CSV...")
    df = pd.read_csv(
        TXT_PATH,
        sep=";",            # semicolon separated
        low_memory=False,   # prevent dtype guessing errors
        na_values="?",      # handle missing values
    )
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved CSV at: {CSV_PATH}")

def cleanup():
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("Deleted ZIP file")

def main():
    download_file()
    unzip_file()
    convert_to_csv()
    cleanup()
    print("All done! Data ready in data/raw/")

if __name__ == "__main__":
    main()
