from __future__ import annotations

import datetime
import typing

import aiosqlite
import attrs
import fastapi

from src.logger import main_logger


@attrs.define
class WSState:
    name: str
    usage: float
    init_time: datetime.datetime

    def get_uptime(self) -> datetime.timedelta:
        return datetime.datetime.now() - self.init_time

    def get_usage(self) -> float:
        """returns kilowatthours of energy used"""
        return self.usage * (self.get_uptime().total_seconds() / 3600)


@attrs.define
class SolarState:
    power_generated: float
    power_consumed: float
    started_at: datetime.datetime
    stopped_at: datetime.datetime


class ConnectionManager:
    sqlite_connection: aiosqlite.Connection

    def __init__(self):
        self.active_connections: dict[fastapi.WebSocket, WSState] = {}

    async def connect(self, websocket: fastapi.WebSocket) -> None:
        await websocket.accept()
        auth = await websocket.receive_json()
        assert (dev := auth.get("device_id")) is not None
        assert (watt := auth.get("usage")) is not None
        self.active_connections[websocket] = (
            state := WSState(dev, watt, datetime.datetime.now())
        )
        main_logger.info(
            f'Device "{state.name}" [{watt}kw] connected to the wesbocket.'
        )

    def disconnect(self, websocket: fastapi.WebSocket) -> None:
        state = self.active_connections.pop(websocket)
        main_logger.info(
            f'Device "{state.name}" [{state.usage}kw] disconnected from the websocket. Connected for {state.get_uptime()}'
        )

    async def broadcast(self, message: str) -> None:
        for connection in self.active_connections:
            await connection.send_text(message)

    async def query_sql(self, query: str, *args: typing.Any) -> None:
        async with self.sqlite_connection.cursor() as cur:
            await cur.execute(query, *args)


manager = ConnectionManager()


async def init_device_websocket_session(ws: fastapi.WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
    except fastapi.WebSocketDisconnect:
        manager.disconnect(ws)
