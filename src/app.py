from __future__ import annotations

import datetime
import typing
from contextlib import asynccontextmanager

import aiosqlite
import fastapi
from src import ws, ml


@asynccontextmanager
async def startup(_: fastapi.FastAPI) -> typing.AsyncGenerator[None]:
    ws.manager.sqlite_connection = await aiosqlite.connect("data.db")
    await ws.manager.query_sql(
        "CREATE TABLE IF NOT EXISTS devices_data ( name VARCHAR, usage FLOAT, uptime FLOAT, started_on TIMESTAMP, stopped_at TIMESTAMP )"
    )
    await ws.manager.query_sql(
        "CREATE TABLE IF NOT EXISTS solar_data ( power_generated FLOAT, power_consumed FLOAT, uptime FLOAT, started_on TIMESTAMP, stopped_at TIMESTAMP)"
    )
    yield
    await ws.manager.sqlite_connection.close()


app = fastapi.FastAPI(lifespan=startup)

start = datetime.datetime.now()


@app.get("/")
def index() -> dict[str, typing.Any]:
    return {
        "status": "online",
        "start_at": start,
        "uptime": datetime.datetime.now() - start,
    }


@app.get("/solardata")
async def get_solor_data() -> dict[str, typing.Any]:
    data = {}
    sessions = await ws.manager.sqlite_connection.execute_fetchall(
        "SELECT * FROM solar_data;"
    )
    prediction = await ml.create_prediction_for_solar_energy(list(sessions))
    total_gen = 0
    total_con = 0
    for session in sessions:
        total_gen += session[0]
        total_con += session[1]
    data["power_generated"] = total_gen
    data["power_consumed"] = total_con
    data["predicted_power"] = float(prediction)
    data["sessions"] = sessions
    return data


@app.post("/solardata")
async def add_solar_data(req: fastapi.Request) -> None:
    data = ws.SolarState(**await req.json())
    await ws.manager.query_sql(
        "INSERT INTO solor_data VALUES (?, ?, ?, ?, ?)",
        data.power_generated,
        data.power_consumed,
        data.stopped_at - data.started_at,
        data.started_at,
        data.stopped_at,
    )

@app.get("/devices_data")
async def get_devices_data() -> dict[str, typing.Any]:
    data = {}
    sessions = await ws.manager.sqlite_connection.execute_fetchall("SELECT * FROM devices_data")
    total_data = list(await ws.manager.sqlite_connection.execute_fetchall("SELECT SUM(usage), SUM(uptime) FROM devices_data"))[0]
    predicted_amount = ml.create_prediction_for_devices(list(sessions))
    data["total_usage"] = total_data[0]
    data["total_uptime"] = total_data[1]
    data["predicted_amount"] = float(predicted_amount)
    data["sessions"] = sessions
    
    return data

app.add_websocket_route("/ws", ws.init_device_websocket_session)
