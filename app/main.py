from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd
import xgboost as xgb

from src.features import create_lag_features, create_rolling_features

app = FastAPI(title="Electricity Consumption Model")

# Load trained model
model_path = "models/model_daily.json"
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained daily model not found. Run run.py first!")

model = xgb.Booster()
model.load_model(model_path)

# Load processed daily dataset
data_path = "data/processed/electricity_daily.csv"
df_daily = pd.read_csv(data_path, parse_dates=["datetime"], index_col="datetime")

# ✅ Ensure continuous daily index
full_range = pd.date_range(df_daily.index.min(), df_daily.index.max(), freq="D")
df_daily = df_daily.reindex(full_range)
df_daily = df_daily.interpolate(method="time", limit_direction="both")
df_daily.index.name = "datetime"

# Input schema
class UserInput(BaseModel):
    date: str   # e.g. "2025-09-01"

@app.get("/")
def home():
    return {"message": "Welcome to Electricity Consumption Model"}

@app.post("/forecast")
def forecast(user_input: UserInput):
    try:
        target_date = pd.to_datetime(user_input.date).normalize()
        last_date = df_daily.index.max()

        if target_date <= last_date:
            return {"error": "Requested date is not in the future. Try a future date."}
        
        df_future = df_daily.copy()
        current_date = last_date

        while current_date < target_date:
            # Add features
            df_features = create_lag_features(df_future.copy())
            df_features = create_rolling_features(df_features)
            df_features = df_features.dropna()

            if df_features.empty:
                return {"error": "Feature engineering failed — empty DataFrame."}

            # Get last row of features
            X_next = df_features.drop("Global_active_power", axis=1).iloc[-1]

            # ✅ Align features with model expectations
            expected_features = model.feature_names
            X_next = X_next.reindex(expected_features, fill_value=0)

            # 🔍 Debugging
            print("=== Prediction Debug ===")
            print("Target date:", target_date)
            print("Expected features:", expected_features)
            print("X_next values:\n", X_next)
            print("========================")

            X_next = X_next.values.reshape(1, -1)

            dmatrix = xgb.DMatrix(X_next, feature_names=expected_features)
            y_pred = model.predict(dmatrix)[0]

            # Add prediction
            next_date = current_date + pd.Timedelta(days=1)
            df_future.loc[next_date] = {"Global_active_power": y_pred}
            current_date = next_date

        forecast_value = float(df_future.loc[target_date, "Global_active_power"])
        return {
            "date": str(target_date.date()),
            "forecast_avg_consumption": forecast_value
        }

    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}
    