import os

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, select, delete, \
    BOOLEAN
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
import datetime
import asyncio
from typing import List, Optional

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

# Очередь задач
task_queue = asyncio.Queue()


# Модель данных
class Stationn(BaseModel):
    id: int
    name: str
    description: str
    created_at: str
    updated_at: Optional[str]
    status: str
    lat: float
    lng: float
    last_temp: Optional[float]
    predictions: Optional[list[float]]
    preds_datetime: Optional[list[str]]
    rules: Optional[list[tuple[int, str]]]
    alerts: Optional[list[str]]


# Модель БД
class Station(Base):
    __tablename__ = "stations"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(String)
    status = Column(String)
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
    active = Column(BOOLEAN)
    station_id = Column(Integer, ForeignKey('stations.id', ondelete='CASCADE'))


class Temperature(Base):
    __tablename__ = "temeratures"
    id = Column(Integer, primary_key=True, index=True)
    value = Column(Float)
    time = Column(DateTime)
    station_id = Column(Integer, ForeignKey('stations.id', ondelete='CASCADE'))


active_connections: List[WebSocket] = []


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
                    Rule(id=0, rule_option='больше', rule_period=2, rule_value=10, station_id=0, active=True),
                    Rule(id=1, rule_option='равно', rule_period=3, rule_value=12, station_id=0, active=True),
                    Rule(id=2, rule_option='меньше', rule_period=1, rule_value=8, station_id=0, active=True),
                    Rule(id=3, rule_option='больше', rule_period=2, rule_value=10, station_id=1, active=True),
                    Rule(id=4, rule_option='равно', rule_period=3, rule_value=12, station_id=1, active=True),
                    Rule(id=5, rule_option='меньше', rule_period=1, rule_value=8, station_id=1, active=True),
                    Rule(id=6, rule_option='больше', rule_period=2, rule_value=10, station_id=2, active=True),
                    Rule(id=7, rule_option='равно', rule_period=3, rule_value=12, station_id=2, active=True),
                    Rule(id=8, rule_option='меньше', rule_period=1, rule_value=8, station_id=2, active=True),
                    Rule(id=9, rule_option='больше', rule_period=2, rule_value=10, station_id=3, active=True),
                    Rule(id=10, rule_option='равно', rule_period=3, rule_value=12, station_id=3, active=True),
                    Rule(id=11, rule_option='меньше', rule_period=1, rule_value=8, station_id=3, active=True),
                    Rule(id=12, rule_option='больше', rule_period=2, rule_value=10, station_id=4),
                    Rule(id=13, rule_option='равно', rule_period=3, rule_value=12, station_id=4),
                    Rule(id=14, rule_option='меньше', rule_period=1, rule_value=8, station_id=4),
                    Rule(id=15, rule_option='больше', rule_period=2, rule_value=10, station_id=5),
                    Rule(id=16, rule_option='равно', rule_period=3, rule_value=12, station_id=5),
                    Rule(id=17, rule_option='меньше', rule_period=1, rule_value=8, station_id=5)
                ])
                await session.commit()


async def get_station_object(station):
    async with SessionLocal() as session:
        async with session.begin():
            # Получаем температуры
            temp_query = await session.execute(
                select(Temperature)
                .where(Temperature.station_id == station.id)
                .order_by(Temperature.time.asc())
            )
            temps = temp_query.scalars().all()

            # Формируем данные
            dates = None
            last_temp = None
            predictions = None

            if temps:
                temps_list = [t.value for t in temps]
                times_list = [t.time.strftime("%Y-%m-%d %H:%M:%S") for t in temps]

                last_temp = temps_list[-1] if temps_list else None
                if len(temps_list) > 1:
                    predictions = temps_list[:-1]
                    dates = times_list[:-1]

            # Получаем правила
            rules_query = await session.execute(
                select(Rule).where(Rule.station_id == station.id)
            )
            rules = rules_query.scalars().all()
            rules_list = [
                (rule.id, f'Температура {rule.rule_option} {rule.rule_value} через {rule.rule_period} часа/часов')
                for rule in rules
            ] if rules else None

            # Формируем алерты
            alerts_list = []
            if rules:
                for rule in rules:
                    if rule.active:
                        alerts_list.append(
                            f'Температура {rule.rule_option} {rule.rule_value} через {rule.rule_period} часа/часов'
                        )
            alerts_list = alerts_list if alerts_list else None

            return Stationn(
                id=station.id,
                name=station.name,
                description=station.description,
                created_at=station.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                updated_at=station.updated_at.strftime("%Y-%m-%d %H:%M:%S") if station.updated_at else None,
                status=station.status,
                lat=station.lat,
                lng=station.lng,
                last_temp=last_temp if station.status == 'В норме' else None,
                predictions=predictions if station.status == 'В норме' else None,
                preds_datetime=dates if station.status == 'В норме' else None,
                rules=rules_list if station.status in ['В норме', 'Выключено'] else None,
                alerts=alerts_list if station.status == 'В норме' else None
            ).model_dump(mode='json')


@app.on_event("startup")
async def startup():
    await init_db()

    # Загрузка прерванных обновляемых станций в очередь
    async with SessionLocal() as session:
        stations = await session.execute(
            select(Station).where(Station.status == 'Обновление')
        )
        for station in stations.scalars().all():
            await task_queue.put(station.id)

    # Проверка устаревших станций
    async with SessionLocal() as session:
        stations = await session.execute(
            select(Station).where(
                Station.updated_at < datetime.datetime.now().replace(minute=0, second=0, microsecond=0))
        )
        for station in stations.scalars().all():
            if station.status == 'В норме':
                station.status = 'Обновление'
                await task_queue.put(station.id)
                await session.commit()


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
async def get_stations():
    async with SessionLocal() as session:
        stations = await session.execute(select(Station))
        return {"stations": {s.id: await get_station_object(s) for s in stations.scalars().all()}}


@app.post("/delete_station")
async def delete_station(station_id: int) -> dict[str, str]:
    async with SessionLocal() as session:
        async with session.begin():
            stations = await session.execute(
                select(Station).where(Station.id == station_id).with_for_update())
            result = stations.scalar_one_or_none()
            if not result:
                await session.rollback()
                message = {
                    "action": "delete",
                    "station_id": station_id,
                    "snack": f"Ошибка: Станции не существует"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}
            if result.status != 'Выключено':
                await session.rollback()
                message = {
                    "action": "update",
                    "station_id": station_id,
                    "values": await get_station_object(result),
                    "snack": f"Ошибка: Перед удалением станция должна быть выключена"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}

            await session.delete(result)
            await session.commit()
            message = {
                "action": "delete",
                "station_id": station_id,
                "snack": f"Станция успешно удалена"
            }
            for conn in active_connections:
                await conn.send_json(message)
            return {"status": "success"}


@app.post("/reload_station")
async def reload_station(station_id: int) -> dict[str, str]:
    async with SessionLocal() as session:
        async with session.begin():
            stations = await session.execute(
                select(Station).where(Station.id == station_id).with_for_update())
            result = stations.scalar_one_or_none()
            if not result:
                await session.rollback()
                message = {
                    "action": "delete",
                    "station_id": station_id,
                    "snack": f"Ошибка: Станции не существует"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}
            if not result.status == 'Ошибка':
                await session.rollback()
                message = {
                    "action": "update",
                    "station_id": station_id,
                    "values": await get_station_object(result),
                    "snack": f"Ошибка: Станция не может быть перезагружена"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}
            result.status = 'Обновление'
            await session.commit()
            await task_queue.put(station_id)
            message = {
                "action": "update",
                "station_id": station_id,
                "values": await get_station_object(result),
                "snack": f"Ошибка: Станция не может быть перезагружена"
            }
            for conn in active_connections:
                await conn.send_json(message)
            return {"status": "success"}


@app.post("/toggle_station")
async def toggle_station(station_id: int) -> dict[str, str]:
    async with SessionLocal() as session:
        async with session.begin():
            stations = await session.execute(
                select(Station).where(Station.id == station_id).with_for_update())
            result = stations.scalar_one_or_none()
            if not result:
                await session.rollback()
                message = {
                    "action": "delete",
                    "station_id": station_id,
                    "snack": f"Ошибка: Станции не существует"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}
            if result.status == 'Ошибка' or result.status == 'В норме':
                result.status = 'Выключено'
                await session.commit()
                message = {
                    "action": "update",
                    "station_id": station_id,
                    "values": await get_station_object(result),
                    "snack": f"Станция успешно выключена"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "success"}
            if result.status == 'Выключено':
                result.status = 'Обновление'
                await task_queue.put(station_id)
                await session.commit()
                message = {
                    "action": "update",
                    "station_id": station_id,
                    "values": await get_station_object(result),
                    "snack": f"Станция успешно выключена"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "success"}


@app.post("/edit_station")
async def edit_station(station_id: int, name: str, descr: str) -> dict[str, str]:
    async with SessionLocal() as session:
        async with session.begin():
            stations = await session.execute(
                select(Station).where(Station.id == station_id).with_for_update())
            result = stations.scalar_one_or_none()
            if not result:
                await session.rollback()
                message = {
                    "action": "delete",
                    "station_id": station_id,
                    "snack": f"Ошибка: Станции не существует"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}

            result.name = name
            result.description = descr
            await session.commit()
            message = {
                "action": "update",
                "station_id": station_id,
                "values": await get_station_object(result),
                "snack": f"Станция успешно отредактирована"
            }

            for conn in active_connections:
                await conn.send_json(message)
            return {"status": "success"}


@app.post("/add_rule")
async def add_rule(station_id: int, rule_option: str, rule_value: str, rule_period: int) -> dict[str, str]:
    async with SessionLocal() as session:
        async with session.begin():

            rules = await session.execute(select(Rule).where(Rule.station_id == station_id))
            if rules and len(rules.scalars().all()) >= 10:
                await session.rollback()
                message = {
                    'action': 'snack',
                    "snack": f"Достигнуто максимальное кол-во правил для этой станции (10)"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}

            stations = await session.execute(
                select(Station).where(Station.id == station_id).with_for_update())
            result = stations.scalar_one_or_none()
            if not result:
                await session.rollback()
                message = {
                    "action": "delete",
                    "station_id": station_id,
                    "snack": f"Ошибка: Станции не существует"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}

            session.add(Rule(rule_option=rule_option, rule_period=rule_period, rule_value=int(rule_value),
                             station_id=station_id, active=False))
            await session.commit()
            await process_rules(station_id)
            message = {
                "action": "update",
                "station_id": station_id,
                "values": await get_station_object(result),
                "snack": f"Правило успешно добавлено"
            }

            for conn in active_connections:
                await conn.send_json(message)

            return {"status": "success"}


@app.post("/add_station")
async def add_station(name: str, descr: str, lat: float, lng: float, activate: bool) -> dict[str, str]:
    async with SessionLocal() as session:
        async with session.begin():
            stations = await session.execute(select(Station))
            if stations and len(stations.scalars().all()) >= 25:
                await session.rollback()
                message = {
                    'action': 'snack',
                    "snack": f"Достигнуто максимальное кол-во станций (25)"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}

            station = Station(name=name, description=descr, created_at=datetime.datetime.now(), updated_at=None,
                              status='Выключено', lat=lat, lng=lng)
            session.add(station)
            await session.commit()

        if activate:
            await task_queue.put(station.id)
            async with session.begin():
                station.status = 'Обновление'
                await session.commit()

        message = {
            "action": "update",
            "station_id": station.id,
            'values': await get_station_object(station),
            "snack": f"Успешно добавлена новая станция"
        }
        for conn in active_connections:
            await conn.send_json(message)

        return {"status": "success"}


@app.post("/delete_rule")
async def delete_rule(station_id: int, rule_id: int) -> dict[str, str]:
    async with SessionLocal() as session:
        async with session.begin():
            stations = await session.execute(
                select(Station).where(Station.id == station_id).with_for_update())
            result = stations.scalar_one_or_none()
            if not result:
                await session.rollback()
                message = {
                    "action": "delete",
                    "station_id": station_id,
                    "snack": f"Ошибка: Станции не существует"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}

            rules = await session.execute(select(Rule).where(Rule.id == rule_id))
            result_rule = rules.scalar_one_or_none()
            if not result:
                await session.rollback()
                message = {
                    "action": "update",
                    "station_id": station_id,
                    "values": await get_station_object(result),
                    "snack": f"Ошибка: Правила не существует"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}

            await session.delete(result_rule)
            await session.commit()
            message = {
                "action": "update",
                "station_id": station_id,
                "values": await get_station_object(result),
                "snack": f"Правило успешно удалена"
            }
            for conn in active_connections:
                await conn.send_json(message)
            return {"status": "success"}


async def process_task(station_id, lat, lng):
    await asyncio.sleep(20)
    try:
        # Обязательно использовать кеш предоставляемый open-meteo sdk
        predictions = [round(i, 4) for i in np.random.uniform(low=-30.0, high=30.0, size=673)],
        preds_datetime = [(datetime.datetime.now() - datetime.timedelta(hours=i)) for i in
                          range(672, -1, -1)],
        return [
            Temperature(
                value=predictions[i],
                time=preds_datetime[i],
                station_id=station_id
            ) for i in range(763)
        ]
    except:
        return 'error'


async def process_rules(station_id):
    async with SessionLocal() as session:
        async with session.begin():

            stations = await session.execute(
                select(Station).where(Station.id == station_id))
            stattion = stations.scalar_one_or_none()
            if not stattion:
                await session.rollback()
                message = {
                    "action": "delete",
                    "station_id": station_id,
                    "snack": f"Ошибка: Станции не существует"
                }
                for conn in active_connections:
                    await conn.send_json(message)
                return {"status": "error"}

            rules = await session.execute(select(Rule).where(Rule.station_id == station_id).with_for_update())
            rules_all = rules.scalars().all()
            if rules_all:
                temperatures = await session.execute(
                    select(Temperature)
                    .where(Temperature.station_id == station_id)
                    .order_by(Temperature.time.desc()).with_for_update()
                )
                query_result = temperatures.scalars().all()
                temps = [temp.value for temp in query_result]
                if not temps:
                    station = await session.execute(select(Station).where(Station.id == station_id))
                    station = station.scalar_one_or_none()
                    station.status = 'Ошибка'
                    session.commit()
                    return
                for rule in rules_all:
                    if rule.rule_option == 'больше':
                        if temps[-rule.rule_period] > rule.rule_value:
                            rule.active = True
                        else:
                            rule.active = False
                    elif rule.rule_option == 'меньше':
                        if temps[-rule.rule_period] < rule.rule_value:
                            rule.active = True
                        else:
                            rule.active = False
                    elif rule.rule_option == 'равна':
                        if temps[-rule.rule_period] == rule.rule_value:
                            rule.active = True
                        else:
                            rule.active = False
                await session.commit()
            else:
                await session.rollback()


async def run_worker():
    while True:
        try:
            station_id = await task_queue.get()
            async with SessionLocal() as session:
                async with session.begin():
                    station = await session.execute(
                        select(Station).where(Station.id == station_id).with_for_update()
                    )
                    station = station.scalar_one_or_none()

                    if station and station.status == 'Обновление':
                        if station.updated_at is None or station.updated_at < datetime.datetime.now().replace(minute=0,
                                                                                                              second=0,
                                                                                                              microsecond=0):
                            try:
                                print(f'Делаем {station_id}')

                                # result_queue = multiprocessing.Queue()
                                # p = multiprocessing.Process(
                                #     target=process_task,
                                #     args=(station_id, station.lat, station.lng, result_queue)
                                # )
                                # p.start()
                                # p.join()
                                # result = result_queue.get()

                                result = await process_task(station_id, station.lat, station.lng)

                                if result == 'error':
                                    station.status = 'Ошибка'
                                    await session.commit()
                                    message = {
                                        "action": "update",
                                        "station_id": station_id,
                                        "values": await get_station_object(station),
                                    }
                                    for conn in active_connections:
                                        await conn.send_json(message)
                                else:
                                    await session.execute(
                                        delete(Temperature).where(Temperature.station_id == station_id)
                                    )
                                    session.add_all(result)
                                    await session.commit()
                                    await task_queue.put(station_id)
                            except:
                                station.status = 'Ошибка'
                                await session.commit()
                        else:
                            station.status = 'В норме'
                            await session.commit()
                            await process_rules(station_id)
                            message = {
                                "action": "update",
                                "station_id": station_id,
                                "values": await get_station_object(station),
                            }
                            for conn in active_connections:
                                await conn.send_json(message)

                    else:
                        await session.rollback()
        except asyncio.QueueEmpty:
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Worker error: {e}")
            await asyncio.sleep(1)


# Запуск сервера
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080)
