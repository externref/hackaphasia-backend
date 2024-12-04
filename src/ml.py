from __future__ import annotations

import typing
import pandas as pd
import numpy as np
from keras import Sequential, layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error

def create_prediction_for_devices(data: list[typing.Any]) -> typing.Any:
    device_data = []
    
    for entry in data:
        device_name, usage, uptime, start_time, stop_time = entry
        start_time = pd.to_datetime(start_time)
        stop_time = pd.to_datetime(stop_time)
        
        device_data.append({
            "name": device_name,
            "usage": usage,
            "uptime": uptime,
            "started_on": start_time,
            "stopped_at": stop_time
        })

    df = pd.DataFrame(device_data)

    df["day_of_week"] = df["started_on"].dt.dayofweek
    df["hour_of_day"] = df["started_on"].dt.hour
    df["month"] = df["started_on"].dt.month
    df["day_of_month"] = df["started_on"].dt.day

    df["name"] = df["name"].astype("category").cat.codes
    df["total_usage"] = df["usage"] * df["uptime"]
    X = df[["name", "uptime", "day_of_week", "hour_of_day", "month", "day_of_month"]]
    y = df["total_usage"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    model = Sequential([
        layers.Dense(64, activation='relu', input_dim=X_train.shape[1]), # type: ignore
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1) # type: ignore

    y_pred = model.predict(X_test)
    # mae = mean_absolute_error(y_test, y_pred)
    next_day_data = np.array([[1, 24, 5, 12, 1, 11]]) 
    next_day_data_scaled = scaler.transform(next_day_data)
    predicted_usage = model.predict(next_day_data_scaled)
    return predicted_usage[0][0] 

