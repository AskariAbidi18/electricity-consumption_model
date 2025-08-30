import pandas as pd
import numpy as np

def create_lag_features(df):
    # daily lags
    df['lag1d'] = df['Global_active_power'].shift(1)
    df['lag2d'] = df['Global_active_power'].shift(2)
    df['lag7d'] = df['Global_active_power'].shift(7)
    return df

def create_rolling_features(df):
    # daily rolling windows
    df['rolling_mean_7d'] = df['Global_active_power'].rolling(window=7).mean()
    df['rolling_mean_30d'] = df['Global_active_power'].rolling(window=30).mean()

    df['rolling_std_7d'] = df['Global_active_power'].rolling(window=7).std()
    df['rolling_std_30d'] = df['Global_active_power'].rolling(window=30).std()

    df['rolling_min_7d'] = df['Global_active_power'].rolling(window=7).min()
    df['rolling_max_7d'] = df['Global_active_power'].rolling(window=7).max()

    return df  

def test_train_split(df):
    X = df.drop('Global_active_power', axis=1)
    y = df['Global_active_power']

    train_size = 0.8
    split_idx = int(len(df) * train_size)

    X_train = X.iloc[:split_idx]    
    y_train = y.iloc[:split_idx]

    X_test  = X.iloc[split_idx:]
    y_test  = y.iloc[split_idx:]

    return X_train, y_train, X_test, y_test
