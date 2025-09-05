import pandas as pd
import numpy as np 

def load_data(path="data/raw/electricity.csv"):
    df = pd.read_csv(
        path,
        low_memory=False
    )
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )
    df = df.drop("Date", axis=1)
    df = df.drop("Time", axis=1)
    df = df.set_index("datetime")
    
    return df

def interpolate(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time",
        limit_direction="both"
    )   
    return df

def resample_daily(df):
    """ Convert minute-level to daily averages with continuous dates """
    df_daily = df.resample("D").mean()

    # Ensure continuous daily index
    full_range = pd.date_range(df_daily.index.min(), df_daily.index.max(), freq="D")
    df_daily = df_daily.reindex(full_range)

    # Interpolate missing values
    df_daily = df_daily.interpolate(method="time", limit_direction="both")

    df_daily.index.name = "datetime"
    return df_daily