import os
import pickle
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Table, ForeignKey, select, delete
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
import requests
import torch
import torch.nn as nn
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import random
import uuid
from typing import List, Dict, Tuple, Optional
from pytorch_forecasting import TemporalFusionTransformer
from config import DATASET_PATH, MODEL_PATH
from fastapi.middleware.cors import CORSMiddleware
from frontend.models import Station as Stationn

# Инициализация приложения
app = FastAPI()

# Конфигурация БД
if os.path.exists('database.db'):
    os.remove('database.db')
DATABASE_URL = "sqlite+aiosqlite:///./database.db"
engine = create_async_engine(DATABASE_URL)
SessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)
Base = declarative_base()


# Модель БД
class Station(Base):
    __tablename__ = "stations"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)
    status = Column(String, index=True)
    lat = Column(Float)
    lng = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.now)
    updated_at = Column(DateTime, default=None)


class Rule(Base):
    __tablename__ = "rules"
    id = Column(Integer, primary_key=True, index=True)
    rule_option = Column(String)
    rule_period = Column(Integer)
    rule_value = Column(Integer)
    station_id = Column(Integer, ForeignKey('stations.id', ondelete='CASCADE'))


class AlertsAudit(Base):
    __tablename__ = "alerts_audit"
    id = Column(Integer, primary_key=True, index=True)
    triggered_at = Column(DateTime)
    description = Column(String)
    rule_id = Column(Integer, ForeignKey("rules.id", ondelete='CASCADE'))


class Temperature(Base):
    __tablename__ = "temeratures"
    id = Column(Integer, primary_key=True, index=True)
    value = Column(Float)
    time = Column(DateTime)
    station_id = Column(Integer, ForeignKey('stations.id', ondelete='CASCADE'))


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Это для примера заполняем различные данные
        async with SessionLocal() as session:
            async with session.begin():
                # Пользователи
                session.add_all([
                    Station(id=0, name="Name 0", description="Description 0",
                            created_at=datetime.datetime.now(),
                            updated_at=datetime.datetime.now(), status='В норме',
                            lat=round(np.random.uniform(-80, 80), 3), lng=round(np.random.uniform(-150, 150), 3)),
                    Station(id=1, name="Name 1", description="Description 1",
                            created_at=datetime.datetime.now(),
                            updated_at=(datetime.datetime.now() - datetime.timedelta(hours=1)), status='В норме',
                            lat=round(np.random.uniform(-80, 80), 3), lng=round(np.random.uniform(-150, 150), 3)),
                    Station(id=2, name="Name 2", description="Description 2",
                            created_at=datetime.datetime.now(),
                            updated_at=None, status='Ошибка',
                            lat=round(np.random.uniform(-80, 80), 3), lng=round(np.random.uniform(-150, 150), 3)),
                    Station(id=3, name="Name 3", description="Description 3",
                            created_at=datetime.datetime.now(),
                            updated_at=(datetime.datetime.now() - datetime.timedelta(hours=2)), status='Обновление',
                            lat=round(np.random.uniform(-80, 80), 3), lng=round(np.random.uniform(-150, 150), 3)),
                    Station(id=4, name="Name 4", description="Description 4",
                            created_at=datetime.datetime.now(),
                            updated_at=(datetime.datetime.now() - datetime.timedelta(hours=2)), status='Выключено',
                            lat=round(np.random.uniform(-80, 80), 3), lng=round(np.random.uniform(-150, 150), 3)),
                    Station(id=5, name="Name 5", description="Description 5",
                            created_at=datetime.datetime.now(),
                            updated_at=datetime.datetime.now(), status='Выключено',
                            lat=round(np.random.uniform(-80, 80), 3), lng=round(np.random.uniform(-150, 150), 3)),
                    Station(id=6, name="Name 6", description="Description 6",
                            created_at=datetime.datetime.now(),
                            updated_at=datetime.datetime.now(), status='Обновление',
                            lat=round(np.random.uniform(-80, 80), 3), lng=round(np.random.uniform(-150, 150), 3)),
                ])

                temps = []
                for st_id in range(7):
                    for i in range(672, -1, -1):
                        temps.append(Temperature(station_id=st_id,
                                                 time=(datetime.datetime.now() - datetime.timedelta(hours=i)),
                                                 value=round(np.random.uniform(low=-30.0, high=30.0), 4)))
                session.add_all(temps)

                session.add_all([
                    Rule(id=0, rule_option='больше', rule_period=2, rule_value=10, station_id=0),
                    Rule(id=1, rule_option='равно', rule_period=3, rule_value=12, station_id=0),
                    Rule(id=2, rule_option='меньше', rule_period=1, rule_value=8, station_id=0),
                    Rule(id=3, rule_option='больше', rule_period=2, rule_value=10, station_id=1),
                    Rule(id=4, rule_option='равно', rule_period=3, rule_value=12, station_id=1),
                    Rule(id=5, rule_option='меньше', rule_period=1, rule_value=8, station_id=1),
                    Rule(id=6, rule_option='больше', rule_period=2, rule_value=10, station_id=2),
                    Rule(id=7, rule_option='равно', rule_period=3, rule_value=12, station_id=2),
                    Rule(id=8, rule_option='меньше', rule_period=1, rule_value=8, station_id=2),
                    Rule(id=9, rule_option='больше', rule_period=2, rule_value=10, station_id=3),
                    Rule(id=10, rule_option='равно', rule_period=3, rule_value=12, station_id=3),
                    Rule(id=11, rule_option='меньше', rule_period=1, rule_value=8, station_id=3),
                    Rule(id=12, rule_option='больше', rule_period=2, rule_value=10, station_id=4),
                    Rule(id=13, rule_option='равно', rule_period=3, rule_value=12, station_id=4),
                    Rule(id=14, rule_option='меньше', rule_period=1, rule_value=8, station_id=4),
                    Rule(id=15, rule_option='больше', rule_period=2, rule_value=10, station_id=5),
                    Rule(id=16, rule_option='равно', rule_period=3, rule_value=12, station_id=5),
                    Rule(id=17, rule_option='меньше', rule_period=1, rule_value=8, station_id=5)
                ])

                session.add_all([
                    AlertsAudit(id=0, rule_id=0, triggered_at=datetime.datetime.now(),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=1, rule_id=1, triggered_at=datetime.datetime.now(),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=2, rule_id=2, triggered_at=datetime.datetime.now(),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=3, rule_id=3, triggered_at=datetime.datetime.now(),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=4, rule_id=4, triggered_at=datetime.datetime.now(),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=5, rule_id=5, triggered_at=datetime.datetime.now(),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=6, rule_id=6, triggered_at=datetime.datetime.now(),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=7, rule_id=7, triggered_at=datetime.datetime.now(),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=8, rule_id=8, triggered_at=datetime.datetime.now(),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=9, rule_id=9, triggered_at=datetime.datetime.now(),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=10, rule_id=10,
                                triggered_at=(datetime.datetime.now() - datetime.timedelta(hours=1)),
                                description='Температура больше 10 через 2 часа'),
                    AlertsAudit(id=11, rule_id=11,
                                triggered_at=(datetime.datetime.now() - datetime.timedelta(hours=1)),
                                description='Температура больше 10 через 2 часа'),
                ])
                await session.commit()


active_connections: List[WebSocket] = []


async def get_station_object(station):
    async with SessionLocal() as session:
        async with session.begin():
            temp_query = await session.execute(
                select(Temperature)
                .where(Temperature.station_id == station.id)
                .order_by(Temperature.time.desc())
            )
            query_result = temp_query.scalars().all()
            dates = None
            last_temp = None
            predictions = None
            if [temp.id for temp in query_result]:
                temps = [temp.value for temp in query_result]
                dates = [temp.time.strftime("%Y-%m-%d %H:%M:%S") for temp in query_result][1:]
                last_temp = temps[0]
                predictions = temps[1:]

            rules_query = await session.execute(
                select(Rule).where(Rule.station_id == station.id)
            )
            rules = rules_query.scalars().all()
            rules_list = [
                (rule.id, f'Температура {rule.rule_option} {rule.rule_value} через {rule.rule_period} часа/часов') for
                rule in rules]
            rules_list = rules_list if rules_list else None

            rule_ids = [rule.id for rule in rules]
            alerts_list = []
            if rule_ids:
                alerts_query = await session.execute(
                    select(AlertsAudit)
                    .where(AlertsAudit.rule_id.in_(rule_ids))
                )
                alerts = alerts_query.scalars().all()
                alerts_list = [alert.description for alert in alerts]
            alerts_list = alerts_list if alerts_list else None

            # Создаем экземпляр dataclass
            station_data = Stationn(
                id=station.id,
                name=station.name,
                description=station.description,
                created_at=station.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                updated_at=station.updated_at.strftime("%Y-%m-%d %H:%M:%S") if station.updated_at else None,
                status=station.status,
                lat=station.lat,
                lng=station.lng,
                last_temp=last_temp if last_temp else None,
                predictions=predictions if predictions else None,
                preds_datetime=dates if dates else None,
                rules=rules_list if rules_list else None,
                alerts=alerts_list if alerts_list else None
            )
            return station_data


@app.on_event("startup")
async def startup():
    await init_db()


# WebSocket эндпоинт
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)


# Эндпоинт для получения данных станций
@app.get("/stations")
async def get_stations() -> dict[str, dict[int, Stationn]]:
    async with SessionLocal() as session:
        # Получаем все станции
        stations = await session.execute(select(Station))
        stations = stations.scalars().all()
        result = {}
        for station in stations:
            result[station.id] = await get_station_object(station)
    return {"stations": result}


@app.post("/delete_station")
async def delete_station(station_id: int) -> dict[str, str]:
    async with SessionLocal() as session:
        async with session.begin():
            stations = await session.execute(select(Station).where(Station.id == station_id))
            result = stations.scalar_one_or_none()
            if not result:
                message = {
                    "action": "update",
                    "station_id": result.id,
                    "values": await get_station_object(result),
                    "snack": f"Ошибка: Станции не существует"
                }
                await session.rollback()
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}
            if not result.status == 'Выключено':
                message = {
                    "action": "update",
                    "station_id": result.id,
                    "values": await get_station_object(result),
                    "snack": f"Ошибка: Перед удалением станция должна быть выключена"
                }
                await session.rollback()
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}
            await session.delete(result)
            message = {
                "action": "delete",
                "station_id": result.id,
                "snack": f"Станция успешно удалена"
            }
            await session.commit()
            for conn in active_connections:
                await conn.send_json(message)
            return {"status": "success"}


# # @app.post("/reload_station")
# # async def reload_station(station_id: int) -> dict[str, str]:
# #     if station_id not in stations_db:
# #         raise HTTPException(
# #             status_code=404,
# #             detail=f"Station {station_id} not found"
# #         )
# #     if stations_db[station_id].status == 'Ошибка':
# #         message = {
# #             "action": "update",
# #             "station_id": station_id,
# #             'values': Station(
# #                 id=station_id,
# #                 name="Normal NoAlerts Station",
# #                 description="Normal NoAlerts meteorological station",
# #                 created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
# #                 updated_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
# #                 status='В норме',
# #                 lat=round(np.random.uniform(-80, 80), 3),
# #                 lng=round(np.random.uniform(-150, 150), 3),
# #                 last_temp=10.534,
# #                 predictions=[round(i, 4) for i in np.random.uniform(low=-30.0, high=30.0, size=672)],
# #                 preds_datetime=[(datetime(2025, 6, 1, hour=5) + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S") for i
# #                                 in
# #                                 range(672)],
# #                 rules=[(i, f'Темпеарутра выше {i}') for i in range(10)],
# #                 alerts=None
# #             ).model_dump(mode='json'),
# #             "snack": f"Станция успешно перезагружена"
# #         }
# #         for conn in active_connections:
# #             await conn.send_json(message)
# #
# #         return {"status": "success"}
# #     else:
# #         message = {
# #             "action": "update",
# #             "station_id": station_id,
# #             'values': stations_db[station_id].model_dump(mode='json'),
# #             "snack": f"Ошибка: Невозможно перезагрузить станцию"
# #         }
# #         for conn in active_connections:
# #             await conn.send_json(message)
# #
# #         return {"status": "error"}
# #
# #
# # @app.post("/toggle_station")
# # async def toggle_station(station_id: int) -> dict[str, str]:
# #     if station_id not in stations_db:
# #         raise HTTPException(
# #             status_code=404,
# #             detail=f"Station {station_id} not found"
# #         )
# #     if stations_db[station_id].status == 'Выключено':
# #         stations_db[station_id] = Station(
# #             id=station_id,
# #             name="Normal NoAlerts Station",
# #             description="Normal NoAlerts meteorological station",
# #             created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
# #             updated_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
# #             status='В норме',
# #             lat=stations_db[station_id].lat,
# #             lng=stations_db[station_id].lng,
# #             last_temp=10.534,
# #             predictions=[round(i, 4) for i in np.random.uniform(low=-30.0, high=30.0, size=672)],
# #             preds_datetime=[(datetime(2025, 6, 1, hour=5) + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S") for i
# #                             in
# #                             range(672)],
# #             rules=[(i, f'Темпеарутра выше {i}') for i in range(10)],
# #             alerts=None
# #         )
# #         message = {
# #             "action": "update",
# #             "station_id": station_id,
# #             'values': stations_db[station_id].model_dump(mode='json'),
# #             "snack": f"Станция успешно включена"
# #         }
# #         for conn in active_connections:
# #             await conn.send_json(message)
# #
# #         return {"status": "success"}
# #     elif stations_db[station_id].status == 'В норме' or stations_db[station_id].status == 'Ошибка':
# #
# #         stations_db[station_id] = Station(
# #             id=station_id,
# #             name="Off Old Station",
# #             description="Off Old meteorological station",
# #             created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
# #             updated_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
# #             status='Выключено',
# #             lat=stations_db[station_id].lat,
# #             lng=stations_db[station_id].lng,
# #             last_temp=10.534,
# #             predictions=None,
# #             preds_datetime=None,
# #             rules=[(i, f'Темпеарутра выше {i}') for i in range(13)],
# #             alerts=None
# #         )
# #         message = {
# #             "action": "update",
# #             "station_id": station_id,
# #             'values': stations_db[station_id].model_dump(mode='json'),
# #             "snack": f"Станция успешно выключена"
# #         }
# #         for conn in active_connections:
# #             await conn.send_json(message)
# #
# #         return {"status": "success"}
# #     else:
# #         message = {
# #             "action": "update",
# #             "station_id": station_id,
# #             'values': stations_db[station_id].model_dump(mode='json'),
# #             "snack": f"Ошибка: Невозможно включить/выключить станцию"
# #         }
# #         for conn in active_connections:
# #             await conn.send_json(message)
# #
# #         return {"status": "error"}
#
#
# @app.post("/edit_station")
# async def edit_station(station_id: int, name: str, descr: str) -> dict[str, str]:
#     if station_id not in stations_db:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Station {station_id} not found"
#         )
#     stations_db[station_id].name = name
#     stations_db[station_id].description = descr
#     message = {
#         "action": "update",
#         "station_id": station_id,
#         'values': stations_db[station_id].model_dump(mode='json'),
#         "snack": f"Станция успешно изменена"
#     }
#     for conn in active_connections:
#         await conn.send_json(message)
#
#     return {"status": "success"}
#
#
# # @app.post("/add_rule")
# # async def add_rule(station_id: int, rule_option: str, rule_value: str, rule_period: int) -> dict[str, str]:
# #     if station_id not in stations_db:
# #         raise HTTPException(
# #             status_code=404,
# #             detail=f"Station {station_id} not found"
# #         )
# #     new_id = len(stations_db[station_id].rules)
# #     stations_db[station_id].rules.append(
# #         (new_id, f'Температура {rule_option} {rule_value} через {rule_period} часа/часов'))
# #
# #     message = {
# #         "action": "update",
# #         "station_id": station_id,
# #         'values': stations_db[station_id].model_dump(mode='json'),
# #         "snack": f"Успешно добавлено правило"
# #     }
# #     for conn in active_connections:
# #         await conn.send_json(message)
# #
# #     return {"status": "success"}
# #
# #
# # @app.post("/add_station")
# # async def add_station(name: str, descr: str, lat: float, lng: float, activate: bool) -> dict[str, str]:
# #     new_key = max(stations_db.keys()) + 1
# #     if activate:
# #         stations_db[new_key] = Station(
# #             id=new_key,
# #             name=name,
# #             description=descr,
# #             created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
# #             updated_at=None,
# #             status='В норме',
# #             lat=lat,
# #             lng=lng,
# #             last_temp=10.534,
# #             predictions=[round(i, 4) for i in np.random.uniform(low=-30.0, high=30.0, size=672)],
# #             preds_datetime=[(datetime(2025, 6, 1, hour=5) + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S") for i
# #                             in
# #                             range(672)],
# #             rules=[(i, f'Темпеарутра выше {i}') for i in range(10)],
# #             alerts=None
# #         )
# #     else:
# #         stations_db[new_key] = Station(
# #             id=new_key,
# #             name=name,
# #             description=descr,
# #             created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
# #             updated_at=None,
# #             status='Выключено',
# #             lat=lat,
# #             lng=lng,
# #             last_temp=None,
# #             predictions=None,
# #             preds_datetime=None,
# #             rules=[(i, f'Темпеарутра выше {i}') for i in range(10)],
# #             alerts=None
# #         )
# #
# #     message = {
# #         "action": "update",
# #         "station_id": new_key,
# #         'values': stations_db[new_key].model_dump(mode='json'),
# #         "snack": f"Успешно добавлена новая станция"
# #     }
# #     for conn in active_connections:
# #         await conn.send_json(message)
# #
# #     return {"status": "success"}
#
#
# @app.post("/delete_rule")
# async def delete_rule(station_id: int, rule_id: int) -> dict[str, str]:
#     if station_id not in stations_db:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Station {station_id} not found"
#         )
#     for data in stations_db[station_id].rules:
#         if data[0] == rule_id:
#             stations_db[station_id].rules.remove(data)
#             break
#
#     message = {
#         "action": "update",
#         "station_id": station_id,
#         'values': stations_db[station_id].model_dump(mode='json'),
#         "snack": f"Успешно удалено правило"
#     }
#     for conn in active_connections:
#         await conn.send_json(message)
#
#     return {"status": "success"}


# Запуск сервера
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)
