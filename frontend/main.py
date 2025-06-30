import asyncio

import numpy as np
from aiohttp import ClientSession, web, ClientConnectionError

from models import Station
from config import RECONNECT_DELAY, WEBSOCKET_IP, STATIONS_API
import flet as ft
from views import StationPage, MainPage
import base64
import datetime
from typing import Dict

with open("assets/logo.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')


class AppManager:
    def __init__(self):
        self.current_page = None
        self.page = None
        self.stations = {}
        self.is_requesting = False

    async def update_station(self, station_id: int, station: Station):
        self.stations[station_id] = station
        if isinstance(self.current_page, StationPage) and self.current_page.station.id == station_id:
            self.current_page.station = self.stations[station_id]
            self.current_page.update()
        else:
            self.current_page.stations = self.stations
            self.current_page.update()

    async def delete_station(self, station_id: int):
        if station_id in self.stations:
            self.stations.pop(station_id)
        if isinstance(self.current_page, StationPage) and self.current_page.station.id == station_id:
            try:
                await self.current_page.close_overlay(None)
            except:
                pass
            await self.current_page.go_back(None)
        else:
            self.current_page.stations = self.stations
            self.current_page.update()


app_manager = AppManager()


def show_main_page(page: ft.Page):
    main_page = MainPage(stations=app_manager.stations, page=page, logo=encoded_string, app=app_manager)
    app_manager.current_page = main_page
    page.controls = [main_page.build()]
    page.update()


async def websocket_client():
    while True:
        try:
            async with ClientSession() as session:
                async with session.ws_connect(WEBSOCKET_IP) as ws:
                    print("WebSocket connected")
                    await load_full_data_from_backend()
                    async for msg in ws:
                        if msg.type == web.WSMsgType.TEXT:
                            data = msg.json()
                            print('Пришло изменение')
                            if data['action'] == 'update':
                                await app_manager.update_station(int(data['station_id']), Station(**data['values']))
                            elif data['action'] == 'delete':
                                await app_manager.delete_station(int(data['station_id']))

                            if data['snack']:
                                app_manager.page.open(
                                    ft.SnackBar(content=ft.Text(value=data['snack']), bgcolor=ft.Colors.ORANGE))

                        elif msg.type == web.WSMsgType.CLOSED:
                            print("WebSocket connection closed")
                            app_manager.page.open(
                                ft.SnackBar(content=ft.Text(value='Соединение к серверу закрыто'), bgcolor=ft.Colors.RED))
                            break
                        elif msg.type == web.WSMsgType.ERROR:
                            print(f"WebSocket error: {ws.exception()}")
                            app_manager.page.open(
                                ft.SnackBar(content=ft.Text(value='Ошибка соединения с сервером'), bgcolor=ft.Colors.RED))
                            break
        except (ClientConnectionError, OSError) as e:
            await session.close()
            app_manager.page.open(
                ft.SnackBar(content=ft.Text(value=f'Ошибка соединения с сервером. Переподключение через {RECONNECT_DELAY} секунд'), bgcolor=ft.Colors.RED))
            print(f"Connection error: {e}. Reconnecting in {RECONNECT_DELAY} seconds...")
        except Exception as e:
            await session.close()
            app_manager.page.open(
                ft.SnackBar(content=ft.Text(value=f'Неизвестная ошибка соединения с сервером'), bgcolor=ft.Colors.RED))
            print(f"Unexpected error: {e}")
        finally:
            await session.close()

        await asyncio.sleep(RECONNECT_DELAY)


async def load_full_data_from_backend():
    # Здесь реализуем загрузку полных данных с бекенда
    try:
        async with ClientSession() as session:
            async with session.get(STATIONS_API) as response:
                if response.status == 200:
                    data = await response.json()
                    # Обновляем локальное состояние
                    app_manager.stations = {int(k): Station(**v) for k, v in data['stations'].items()}
                response.close()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await session.close()


async def main(page: ft.Page):
    page.theme_mode = ft.ThemeMode.LIGHT
    page.browser_context_menu.disable()
    app_manager.page = page
    await load_full_data_from_backend()
    show_main_page(page=page)
    page.run_task(websocket_client)


if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, assets_dir="/assets")
