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

def create_prediction_for_solar_energy(data: list[typing.Any]) -> typing.Any:
    solar_data = []
    
    for entry in data:
        power_generated, power_consumed, uptime, started_on, stopped_at = entry
        started_on = pd.to_datetime(started_on)
        stopped_at = pd.to_datetime(stopped_at)
        
        solar_data.append({
            "power_generated": power_generated,
            "power_consumed": power_consumed,
            "uptime": uptime,
            "started_on": started_on,
            "stopped_at": stopped_at
        })

    df = pd.DataFrame(solar_data)

    df["hour_of_day"] = df["started_on"].dt.hour
    df["month"] = df["started_on"].dt.month
    df["day_of_month"] = df["started_on"].dt.day

    df["total_uptime"] = (df["stopped_at"] - df["started_on"]).dt.total_seconds() / 3600  # Convert to hours

    df["power_balance"] = df["power_generated"] - df["power_consumed"]

    X = df[["power_generated", "power_consumed", "uptime", "day_of_week", "hour_of_day", "month", "day_of_month"]]
    y = df["power_balance"]

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

    next_day_data = np.array([[8.5, 6.2, 12, 3, 14, 6, 12]]) 
    next_day_data_scaled = scaler.transform(next_day_data)
    predicted_balance = model.predict(next_day_data_scaled)

    return predicted_balance[0][0]
