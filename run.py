from src.download_data import main as download_raw
from src.data_preprocessing import load_data, interpolate, resample_daily
from src.features import create_lag_features, create_rolling_features, test_train_split
from src.model import xgBoost
from src.evaluate import evaluate_model
import os

def main():
    PROCESSED_PATH = "data/processed/electricity_daily.csv"

    # 1. Download raw data
    download_raw()

    # 2. Load & preprocess
    df = load_data()
    df = interpolate(df)

    # 3. Resample to daily averages
    df_daily = resample_daily(df)

    # 4. Feature engineering
    df_daily = create_lag_features(df_daily)
    df_daily = create_rolling_features(df_daily)
    df_daily = df_daily.dropna()

    # 5. Save processed daily data
    os.makedirs("data/processed", exist_ok=True)
    df_daily.to_csv(PROCESSED_PATH)
    print(f"Daily processed data saved at {PROCESSED_PATH}")

    # 6. Train/test split
    X_train, y_train, X_test, y_test = test_train_split(df_daily)

    # 7. Train & save model
    model, y_pred = xgBoost(X_train, y_train, X_test, y_test)
    os.makedirs("models", exist_ok=True)
    model.save_model("models/model_daily.json")

    # 8. Evaluate
    print("Daily Model Evaluation - \n")
    evals = evaluate_model(y_test, y_pred)
    for i in evals:
        print(i, " : ", evals[i])

if __name__ == "__main__":
    main()
