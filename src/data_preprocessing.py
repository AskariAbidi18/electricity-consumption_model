import pandas as pd
import numpy as np 

def load_data():
    df = pd.read_csv(
    "electricity.csv",
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

def interpolate():
    df = load_data()
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df_interp[numeric_cols].interpolate(
    method="time",
    limit_direction="both"
    )   

    return df


