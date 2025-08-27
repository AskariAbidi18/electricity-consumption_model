import pandas as pd
import numpy as np

def create_lag_features():
    df['lag1h'] = df['Global_active_power'].shift(60)
    df['lag2h'] = df['Global_active_power'].shift(120)
    df['lag3h'] = df['Global_active_power'].shift(180)
    df['lag24h'] = df['Global_active_power'].shift(24*60)
    df['lag7d'] = df['Global_active_power'].shift(24*60*7)

    return df 

def create_rolling_features():
    df['rolling_mean_1h'] = df['Global_active_power'].rolling(window=60).mean()
    df['rolling_mean_24h'] = df['Global_active_power'].rolling(window=1440).mean()
    df['rolling_mean_7d'] = df['Global_active_power'].rolling(window=10080).mean()
    df['rolling_std_1h'] = df['Global_active_power'].rolling(window=60).std()
    df['rolling_min_1h'] = df['Global_active_power'].rolling(window=60).min()
    df['rolling_max_1h'] = df['Global_active_power'].rolling(window=60).max()
    df['rolling_std_24h'] = df['Global_active_power'].rolling(window=1440).std()
    df['rolling_min_24h'] = df['Global_active_power'].rolling(window=1440).min()
    df['rolling_max_24h'] = df['Global_active_power'].rolling(window=1440).max()
    df['rolling_std_7d'] = df['Global_active_power'].rolling(window=10080).std()
    df['rolling_min_7d'] = df['Global_active_power'].rolling(window=10080).min()
    df['rolling_max_7d'] = df['Global_active_power'].rolling(window=10080).max()

    return df  

def test_train_split():
    X = df.drop('Global_active_power', axis=1)
    y = df['Global_active_power']

    train_size = 0.8
    split_idx = int(len(df) * train_size)

    X_train = X.iloc[:split_idx]    
    y_train = y.iloc[:split_idx]

    X_test  = X.iloc[split_idx:]
    y_test  = y.iloc[split_idx:]

    split = [X_train,y_train,X_test,y_test]

    return split
