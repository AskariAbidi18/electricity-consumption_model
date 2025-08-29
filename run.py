from src.download_data import main as download_raw
from src.data_preprocessing import load_data, interpolate
from src.features import create_lag_features, create_rolling_features, test_train_split
from src.model import xgBoost
from src.evaluate import evaluate_model
import pandas as pd
import os

def main():

    PROCESSED_PATH = "data/processed/electricity.csv"

    #1. Downlading the raw data
    download_raw()

    #2. data preprocessing
    df = load_data()

    df = interpolate()

    #3. feature engineering

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = df.dropna()

    #5. save the processed data

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH)
    print(f"Processed data saved at {PROCESSED_PATH}")

    #test train split 

    X_train, y_train, X_test, y_test = test_train_split(df)

    #6. training and saving model

    model, y_pred = xgBoost(X_train, y_train, X_test, y_test)
    os.makedirs("models", exist_ok=True)
    model.save_model("models/model.json")

    #7. model evaluation 

    print("Model Evaluation - \n")
    evals = evaluate_model(y_test, y_pred)
    for i in evals:
        print(i, " : ", evals[i])

if __name__ == "__main__":
    main()