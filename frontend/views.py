import asyncio
import datetime

import numpy as np
import flet_map as map
from aiohttp import ClientSession

from models import Station
import flet as ft
from config import DELETE_URL, RELOAD_URL, TOGGLE_URL, DELETE_RULE_URL, ADD_RULE_URL, ADD_STATION_URL, EDIT_STATION_URL


class StationPage:
    def __init__(self, page: ft.Page, station: Station, logo, app):
        self.station_newname = None
        self.station_newdesc = None
        self.chart_container = None
        self.active_alerts = None
        self.labels_bottom = None
        self.pts = None
        self.alert_rules = None
        self.control_row = None
        self.name_description_row = None
        self.info_row = ft.Row()
        self.station = station
        self.page = page
        self.chart = None
        self.info_text = None
        self.viewport = {"min": 0, "width": 36}
        self.logo = logo
        self.rule_option = None
        self.app_manager = app

        self.rule_sign = None
        self.rule_value = None
        self.rule_period = None

    def _get_close(self, num, coef):
        return int(round(num / coef) * coef if num >= 0 else -(round(abs(num) / coef) * coef))

    def build(self):
        if isinstance(self.station.created_at, str):
            self.station.created_at = datetime.datetime.strptime(self.station.created_at, "%Y-%m-%d %H:%M:%S")
        if self.station.preds_datetime:
            if isinstance(self.station.preds_datetime[0], str):
                self.station.preds_datetime = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in
                                               self.station.preds_datetime]
        if self.station.updated_at and isinstance(self.station.updated_at, str):
            self.station.updated_at = datetime.datetime.strptime(self.station.updated_at, "%Y-%m-%d %H:%M:%S")

        status = self.station.status
        power_icon = ft.Icons.POWER_OFF
        toggle_disable = False
        delete_disable = True
        reload_disable = True

        power_text_color = ft.Colors.BLACK
        power_text = 'Выключить'
        power_color = ft.Colors.RED
        delete_icon_color = ft.Colors.GREY
        delete_text_color = ft.Colors.GREY
        reload_icon_color = ft.Colors.GREY
        reload_text_color = ft.Colors.GREY
        if status == 'В норме':
            icon = ft.Icons.CHECK_CIRCLE
            icon_color = ft.Colors.GREEN
        elif status == 'Выключено':
            icon = ft.Icons.OFFLINE_PIN
            icon_color = ft.Colors.GREY
            power_icon = ft.Icons.POWER
            power_color = ft.Colors.GREEN
            delete_disable = False
            power_text = 'Включить'
            delete_text_color = ft.Colors.BLACK
        elif status == 'Обновление':
            icon = ft.Icons.UPDATE
            icon_color = ft.Colors.YELLOW
            power_color = ft.Colors.GREY
            power_text_color = ft.Colors.GREY
            toggle_disable = True
        else:
            icon = ft.Icons.ERROR
            icon_color = ft.Colors.RED
            reload_icon_color = ft.Colors.BLUE
            reload_text_color = ft.Colors.BLACK
            reload_disable = False

        self.chart_container = ft.Container(height=450)

        # Ряд с информацией
        self.info_row = ft.Row(controls=[

            # Статус
            ft.Row(controls=[
                ft.Text(value='Статус:', color=ft.Colors.BLACK),
                ft.Icon(name=icon, color=icon_color, size=20),
                ft.Text(value=status, color=ft.Colors.BLACK),
            ], spacing=1),

            # Обновлено
            ft.Row(controls=[
                ft.Text(value='Обновлено:', color=ft.Colors.BLACK),
                ft.Text(value=str(self.station.updated_at) if self.station.updated_at else 'Никогда',
                        color=ft.Colors.BLACK),
            ], spacing=1),

            # Температура
            ft.Row(controls=[
                ft.Text(value='🌡️', color=ft.Colors.RED, size=20),
                ft.Text(value=f'{round(self.station.last_temp, 1)}°C', color=ft.Colors.BLACK),
            ], spacing=1) if status == 'В норме' else ft.Row(controls=[
                ft.Text(value='🌡️', color=ft.Colors.RED, size=20),
                ft.Text(value=f'--°C', color=ft.Colors.BLACK),
            ], spacing=1),

            # Координаты
            ft.Row(controls=[
                ft.Icon(name=ft.Icons.LOCATION_ON, color=ft.Colors.RED, size=20),
                ft.Text(value=f'{self.station.lat} ш., {self.station.lng} д.', color=ft.Colors.BLACK),
            ], spacing=1),

            # Добавлено
            ft.Row(controls=[
                ft.Text(value='Добавлено:', color=ft.Colors.BLACK),
                ft.Text(value=str(self.station.created_at), color=ft.Colors.BLACK),
            ], spacing=1),
        ], spacing=20, height=30)

        # Ряд с названием и описанием
        self.name_description_row = ft.Row(controls=[
            ft.IconButton(icon=ft.Icons.EDIT, on_click=self.edit_station, icon_size=20,
                          alignment=ft.alignment.center_left),
            ft.Text(value=f'Название: {self.station.name}', color=ft.Colors.BLACK, text_align=ft.TextAlign.CENTER),
            ft.Text(value=f'Описание: {self.station.description}', color=ft.Colors.BLACK,
                    text_align=ft.TextAlign.CENTER)
        ], spacing=18, height=30)

        # Ряд с управлением станцией
        self.control_row = ft.Row(controls=[
            ft.ElevatedButton(icon=power_icon, icon_color=power_color, disabled=toggle_disable, text=power_text,
                              color=power_text_color, on_click=self.toggle_station),
            ft.ElevatedButton(icon=ft.Icons.DELETE_FOREVER, icon_color=delete_icon_color, text='Удалить',
                              color=delete_text_color, disabled=delete_disable, on_click=self.station_delete),
            ft.ElevatedButton(icon=ft.Icons.UPDATE, text='Перезагрузить', disabled=reload_disable,
                              icon_color=reload_icon_color, color=reload_text_color, on_click=self.station_reload)
        ], spacing=20, height=30)

        # График
        if self.station.predictions and self.station.preds_datetime:
            x = self.station.preds_datetime
            y = self.station.predictions

            self.pts = [ft.LineChartDataPoint(x=i, y=y_) for i, y_ in enumerate(y)]

            self.labels_bottom = [
                ft.ChartAxisLabel(value=i, label=ft.Column(controls=[
                    ft.Text(x[i].strftime("%y-%m-%d"), size=10, max_lines=1),
                    ft.Text('   ' + x[i].strftime("%H:%M"), size=10, max_lines=1)
                ], spacing=0))
                for i in range(0, 672)
            ]

            self.chart = ft.LineChart(
                data_series=[ft.LineChartData(data_points=self.pts[0:36], color=ft.Colors.RED, stroke_width=2)],
                left_axis=ft.ChartAxis(
                    labels=[
                               ft.ChartAxisLabel(
                                   value=i,
                                   label=ft.Text(str(int(i)), size=10, no_wrap=True, max_lines=1)
                               ) for i in
                               range(self._get_close(min(y), 10) + 10, self._get_close(max(y), 10) - 10 + 1, 10)
                           ] + [ft.ChartAxisLabel(value=min(y),
                                                  label=ft.Text(str(round(min(y), 1)), size=10, no_wrap=True,
                                                                max_lines=1)),
                                ft.ChartAxisLabel(value=max(y),
                                                  label=ft.Text(str(round(max(y), 1)), size=10, no_wrap=True,
                                                                max_lines=1))]
                ),
                bottom_axis=ft.ChartAxis(labels=self.labels_bottom[0:36]),
                min_y=min(y),
                max_y=max(y),
                interactive=True,
                animate=0,
                horizontal_grid_lines=ft.ChartGridLines(),
                vertical_grid_lines=ft.ChartGridLines(),
                height=450,
                tooltip_bgcolor=ft.Colors.WHITE
            )
            self.chart.bottom_axis.labels[0].visible = False
            self.chart.bottom_axis.labels[-1].visible = False

            self.chart_container.content = ft.GestureDetector(
                content=self.chart,
                on_scroll=self.on_scroll, height=450
            )
        else:
            ttext = 'Нет данных'
            if status == 'Обновление':
                ttext = 'Обновление данных'
            if status == 'Ошибка':
                ttext = 'Ошибка'

            self.chart = ft.LineChart(
                left_axis=ft.ChartAxis(),
                interactive=True,
                animate=0,
                horizontal_grid_lines=ft.ChartGridLines(),
                vertical_grid_lines=ft.ChartGridLines(),
                height=450,
                tooltip_bgcolor=ft.Colors.WHITE
            )
            self.chart_container.content = ft.Text(value=ttext, color=ft.Colors.BLACK, height=450, expand=True, size=40)

        if self.station.rules:
            self.alert_rules = ft.Column(controls=[
                ft.Row(controls=[
                    ft.Row(controls=[
                        ft.Icon(name=ft.Icons.ALARM, color=ft.Colors.YELLOW_800),
                        ft.Text(value=f'Правила алертов ({len(self.station.rules)})', color=ft.Colors.BLACK)
                    ], spacing=0),
                    ft.ElevatedButton(text='+ Создать новое правило', color=ft.Colors.BLACK,
                                      on_click=self.add_new_rule, disabled=len(self.station.rules) >= 10)
                ], height=30, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Column(scroll=ft.ScrollMode.ALWAYS, controls=[
                    ft.Row(controls=[
                        ft.Text(value=name, color=ft.Colors.BLACK),
                        ft.IconButton(icon=ft.Icons.DELETE, on_click=lambda e, i=idd: self.delete_rule_st(i))
                    ], spacing=5) for idd, name in self.station.rules
                ], expand=True)
            ], expand=True)
        else:
            ttext = 'Нет правил, создайте одно'
            dis = False
            if status == 'Обновление':
                ttext = 'Обновление данных'
                dis = True
            if status == 'Ошибка':
                ttext = 'Ошибка'
                dis = True
            self.alert_rules = ft.Column(controls=[
                ft.Row(controls=[
                    ft.Row(controls=[
                        ft.Icon(name=ft.Icons.ALARM, color=ft.Colors.YELLOW_800),
                        ft.Text(value=f'Правила алертов (0)', color=ft.Colors.BLACK)
                    ], spacing=0),
                    ft.ElevatedButton(text='+ Создать новое правило', color=ft.Colors.BLACK, on_click=self.add_new_rule,
                                      disabled=dis)
                ], height=30, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text(value=ttext, color=ft.Colors.BLACK, expand=True)
            ], expand=True)

        if self.station.alerts:
            self.active_alerts = ft.Column(controls=[
                ft.Row(controls=[
                    ft.Icon(name=ft.Icons.WARNING, color=ft.Colors.YELLOW_800),
                    ft.Text(value=f'Активные алерты ({len(self.station.alerts)})', color=ft.Colors.BLACK)
                ], height=30),
                ft.Column(scroll=ft.ScrollMode.ALWAYS, controls=[
                    ft.Row(controls=[
                        ft.Text(value=name, color=ft.Colors.BLACK)
                    ]) for name in self.station.alerts
                ], expand=True)
            ], expand=True)
        else:
            ttext = 'Нет активных алертов'
            if status == 'Обновление':
                ttext = 'Обновление данных'
            if status == 'Ошибка':
                ttext = 'Ошибка'
            if status == 'Выключено':
                ttext = 'Нет данных'
            self.active_alerts = ft.Column(controls=[
                ft.Row(controls=[
                    ft.Icon(name=ft.Icons.WARNING, color=ft.Colors.YELLOW_800),
                    ft.Text(value=f'Активные алерты (0)', color=ft.Colors.BLACK)
                ], height=30),
                ft.Text(value=ttext, color=ft.Colors.BLACK, expand=True)
            ], expand=True)

        return ft.Container(expand=True,
                            gradient=ft.LinearGradient(colors=[ft.Colors.WHITE, ft.Colors.LIGHT_BLUE_100],
                                                       begin=ft.alignment.top_center,
                                                       end=ft.alignment.bottom_center),
                            content=ft.Column(controls=[
                                ft.Row(
                                    controls=[
                                        ft.Image(src_base64=self.logo, fit=ft.ImageFit.CONTAIN, width=60,
                                                 height=60),
                                        ft.Text('Feyra weather', color=ft.Colors.BLACK,
                                                weight=ft.FontWeight.BOLD,
                                                size=40),
                                    ]
                                ),
                                ft.Container(content=ft.Row(controls=[
                                    ft.ElevatedButton(width=140, height=30, text='НА ГЛАВНУЮ',
                                                      color=ft.Colors.BLACK,
                                                      icon=ft.Icons.CLOSE,
                                                      icon_color=ft.Colors.RED, on_click=self.go_back),
                                    self.info_row
                                ], spacing=30, height=40, expand=True), border_radius=8,
                                    bgcolor=ft.Colors.LIGHT_BLUE_100, padding=5,
                                    shadow=ft.BoxShadow(spread_radius=3, blur_radius=3,
                                                        color=ft.Colors.LIGHT_BLUE_100)),
                                ft.Container(content=self.name_description_row, height=40, border_radius=8,
                                             bgcolor=ft.Colors.LIGHT_BLUE_100,
                                             shadow=ft.BoxShadow(spread_radius=3, blur_radius=3,
                                                                 color=ft.Colors.LIGHT_BLUE_100)),
                                ft.Container(content=self.control_row, height=40, border_radius=8,
                                             bgcolor=ft.Colors.LIGHT_BLUE_100, padding=5,
                                             shadow=ft.BoxShadow(spread_radius=3, blur_radius=3,
                                                                 color=ft.Colors.LIGHT_BLUE_100)),
                                self.chart_container,
                                ft.Row(controls=[
                                    ft.Container(content=self.alert_rules, expand=True,
                                                 border=ft.border.all(width=1, color=ft.Colors.BLUE), padding=5,
                                                 border_radius=8),
                                    ft.Container(content=self.active_alerts, expand=True,
                                                 border=ft.border.all(width=1, color=ft.Colors.BLUE), padding=5,
                                                 border_radius=8)
                                ], expand=True)
                            ], expand=True, spacing=10))

    def delete_rule_st(self, rule_id):
        asyncio.run(self.delete_rule(rule_id))

    def on_scroll(self, e: ft.ScrollEvent):
        if e.scroll_delta_y:
            dx = int(np.sign(e.scroll_delta_y) * 1)  # (viewport["width"] * 0.1)
            new_min = min(max(0, self.viewport["min"] + dx), 672 - self.viewport["width"])
            self.viewport["min"] = new_min
            new_max = new_min + self.viewport["width"]
            self.chart.data_series = [
                ft.LineChartData(data_points=self.pts[new_min:new_max], color=ft.Colors.RED, stroke_width=2)]
            lbl = self.labels_bottom[int(new_min):int(new_max)]
            lbl[0].visible = False
            lbl[-1].visible = False
            self.chart.bottom_axis.labels = lbl
            self.chart.update()
            lbl[0].visible = True
            lbl[-1].visible = True

    def update(self):
        if isinstance(self.station.created_at, str):
            self.station.created_at = datetime.datetime.strptime(self.station.created_at, "%Y-%m-%d %H:%M:%S")
        if self.station.preds_datetime:
            if isinstance(self.station.preds_datetime[0], str):
                self.station.preds_datetime = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in
                                               self.station.preds_datetime]
        if self.station.updated_at and isinstance(self.station.updated_at, str):
            self.station.updated_at = datetime.datetime.strptime(self.station.updated_at, "%Y-%m-%d %H:%M:%S")

        self.viewport = {"min": 0, "width": 36}
        if self.station.rules:
            self.alert_rules.controls = [
                ft.Row(controls=[
                    ft.Row(controls=[
                        ft.Icon(name=ft.Icons.ALARM, color=ft.Colors.YELLOW_800),
                        ft.Text(value=f'Правила алертов ({len(self.station.rules)})', color=ft.Colors.BLACK)
                    ], spacing=0),
                    ft.ElevatedButton(text='+ Создать новое правило', color=ft.Colors.BLACK, on_click=self.add_new_rule,
                                      disabled=len(self.station.rules) >= 10)
                ], height=30, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Column(scroll=ft.ScrollMode.ALWAYS, controls=[
                    ft.Row(controls=[
                        ft.Text(value=name, color=ft.Colors.BLACK),
                        ft.IconButton(icon=ft.Icons.DELETE, on_click=lambda e, i=idd: self.delete_rule_st(i))
                    ], spacing=5) for idd, name in self.station.rules
                ], expand=True)
            ]
        else:
            ttext = 'Нет правил, создайте одно'
            dis = False
            if self.station.status == 'Обновление':
                ttext = 'Обновление данных'
                dis = True
            if self.station.status == 'Ошибка':
                ttext = 'Ошибка'
                dis = True

            self.alert_rules.controls = [
                ft.Row(controls=[
                    ft.Row(controls=[
                        ft.Icon(name=ft.Icons.ALARM, color=ft.Colors.YELLOW_800),
                        ft.Text(value=f'Правила алертов (0)', color=ft.Colors.BLACK)
                    ], spacing=0),
                    ft.ElevatedButton(text='+ Создать новое правило', color=ft.Colors.BLACK, on_click=self.add_new_rule,
                                      disabled=dis)
                ], height=30, alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text(value=ttext, color=ft.Colors.BLACK, expand=True)
            ]

        if self.station.alerts:
            self.active_alerts.controls = [
                ft.Row(controls=[
                    ft.Icon(name=ft.Icons.WARNING, color=ft.Colors.YELLOW_800),
                    ft.Text(value=f'Активные алерты ({len(self.station.alerts)})', color=ft.Colors.BLACK)
                ], height=30),
                ft.Column(scroll=ft.ScrollMode.ALWAYS, controls=[
                    ft.Row(controls=[
                        ft.Text(value=name, color=ft.Colors.BLACK)
                    ]) for name in self.station.alerts
                ], expand=True, alignment=ft.MainAxisAlignment.START)
            ]
        else:
            ttext = 'Нет активных алертов'
            if self.station.status == 'Обновление':
                ttext = 'Обновление данных'
            if self.station.status == 'Ошибка':
                ttext = 'Ошибка'
            if self.station.status == 'Выключено':
                ttext = 'Нет данных'
            self.active_alerts.controls = [
                ft.Row(controls=[
                    ft.Icon(name=ft.Icons.WARNING, color=ft.Colors.YELLOW_800),
                    ft.Text(value=f'Активные алерты (0)', color=ft.Colors.BLACK)
                ], height=30),
                ft.Text(value=ttext, color=ft.Colors.BLACK, expand=True)
            ]

        if self.station.predictions and self.station.preds_datetime:
            x = self.station.preds_datetime
            y = self.station.predictions

            self.pts = [ft.LineChartDataPoint(x=i, y=y_) for i, y_ in enumerate(y)]

            self.labels_bottom = [
                ft.ChartAxisLabel(value=i, label=ft.Column(controls=[
                    ft.Text(x[i].strftime("%y-%m-%d"), size=10, max_lines=1),
                    ft.Text('   ' + x[i].strftime("%H:%M"), size=10, max_lines=1)
                ], spacing=0))
                for i in range(0, 672)
            ]
            self.chart.data_series = [ft.LineChartData(data_points=self.pts[0:36], color=ft.Colors.RED, stroke_width=2)]
            self.chart.left_axis.labels = [
                                              ft.ChartAxisLabel(
                                                  value=i,
                                                  label=ft.Text(str(int(i)), size=10, no_wrap=True, max_lines=1)
                                              ) for i in
                                              range(self._get_close(min(y), 10) + 10,
                                                    self._get_close(max(y), 10) - 10 + 1, 10)
                                          ] + [ft.ChartAxisLabel(value=min(y),
                                                                 label=ft.Text(str(round(min(y), 1)), size=10,
                                                                               no_wrap=True,
                                                                               max_lines=1)),
                                               ft.ChartAxisLabel(value=max(y),
                                                                 label=ft.Text(str(round(max(y), 1)), size=10,
                                                                               no_wrap=True,
                                                                               max_lines=1))]
            self.chart.bottom_axis = ft.ChartAxis(labels=self.labels_bottom[0:36])
            self.chart.bottom_axis.labels[0].visible = False
            self.chart.bottom_axis.labels[-1].visible = False
            self.chart.min_y = min(y)
            self.chart.max_y = max(y)

            if isinstance(self.chart_container.content, ft.GestureDetector):
                self.chart_container.content.content = self.chart
            else:
                self.chart_container.content = ft.GestureDetector(
                    content=self.chart,
                    on_scroll=self.on_scroll, height=450
                )
        else:
            ttext = 'Нет данных'
            if self.station.status == 'Обновление':
                ttext = 'Обновление данных'
            if self.station.status == 'Ошибка':
                ttext = 'Ошибка'
            self.chart_container.content = ft.Text(value=ttext, color=ft.Colors.BLACK, height=450, expand=True, size=40)

        status = self.station.status
        power_icon = ft.Icons.POWER_OFF
        toggle_disable = False
        delete_disable = True
        reload_disable = True
        power_text_color = ft.Colors.BLACK
        power_text = 'Выключить'
        power_color = ft.Colors.RED
        delete_icon_color = ft.Colors.GREY
        delete_text_color = ft.Colors.GREY
        reload_icon_color = ft.Colors.GREY
        reload_text_color = ft.Colors.GREY
        if status == 'В норме':
            icon = ft.Icons.CHECK_CIRCLE
            icon_color = ft.Colors.GREEN
        elif status == 'Выключено':
            icon = ft.Icons.OFFLINE_PIN
            icon_color = ft.Colors.GREY
            power_icon = ft.Icons.POWER
            power_color = ft.Colors.GREEN
            delete_disable = False
            power_text = 'Включить'
            delete_text_color = ft.Colors.BLACK
        elif status == 'Обновление':
            icon = ft.Icons.UPDATE
            icon_color = ft.Colors.YELLOW
            power_color = ft.Colors.GREY
            power_text_color = ft.Colors.GREY
            toggle_disable = True
        else:
            icon = ft.Icons.ERROR
            icon_color = ft.Colors.RED
            reload_icon_color = ft.Colors.BLUE
            reload_text_color = ft.Colors.BLACK
            reload_disable = False

        # Ряд с информацией
        self.info_row.controls = [

            # Статус
            ft.Row(controls=[
                ft.Text(value='Статус:', color=ft.Colors.BLACK),
                ft.Icon(name=icon, color=icon_color, size=20),
                ft.Text(value=status, color=ft.Colors.BLACK),
            ], spacing=1),

            # Обновлено
            ft.Row(controls=[
                ft.Text(value='Обновлено:', color=ft.Colors.BLACK),
                ft.Text(value=str(self.station.updated_at) if self.station.updated_at else 'Никогда',
                        color=ft.Colors.BLACK),
            ], spacing=1),

            # Температура
            ft.Row(controls=[
                ft.Text(value='🌡️', color=ft.Colors.RED, size=20),
                ft.Text(value=f'{round(self.station.last_temp, 1)}°C', color=ft.Colors.BLACK),
            ], spacing=1) if status == 'В норме' else ft.Row(controls=[
                ft.Text(value='🌡️', color=ft.Colors.RED, size=20),
                ft.Text(value=f'--°C', color=ft.Colors.BLACK),
            ], spacing=1),

            # Координаты
            ft.Row(controls=[
                ft.Icon(name=ft.Icons.LOCATION_ON, color=ft.Colors.RED, size=20),
                ft.Text(value=f'{self.station.lat} ш., {self.station.lng} д.', color=ft.Colors.BLACK),
            ], spacing=1),

            # Добавлено
            ft.Row(controls=[
                ft.Text(value='Добавлено:', color=ft.Colors.BLACK),
                ft.Text(value=str(self.station.created_at), color=ft.Colors.BLACK),
            ], spacing=1),
        ]

        # Ряд с названием и описанием
        self.name_description_row.controls = [
            ft.IconButton(icon=ft.Icons.EDIT, on_click=self.edit_station, icon_size=20,
                          alignment=ft.alignment.center_left),
            ft.Text(value=f'Название: {self.station.name}', color=ft.Colors.BLACK, text_align=ft.TextAlign.CENTER),
            ft.Text(value=f'Описание: {self.station.description}', color=ft.Colors.BLACK,
                    text_align=ft.TextAlign.CENTER)
        ]

        # Ряд с управлением станцией
        self.control_row.controls = [
            ft.ElevatedButton(icon=power_icon, icon_color=power_color, disabled=toggle_disable, text=power_text,
                              color=power_text_color, on_click=self.toggle_station),
            ft.ElevatedButton(icon=ft.Icons.DELETE_FOREVER, icon_color=delete_icon_color, text='Удалить',
                              color=delete_text_color, disabled=delete_disable, on_click=self.station_delete),
            ft.ElevatedButton(icon=ft.Icons.UPDATE, text='Перезагрузить', disabled=reload_disable,
                              icon_color=reload_icon_color, color=reload_text_color, on_click=self.station_reload)
        ]

        self.page.update()

    async def go_back(self, e):
        main_page = MainPage(stations=self.app_manager.stations, page=self.page, logo=self.logo, app=self.app_manager)
        self.app_manager.current_page = main_page
        self.page.controls = [main_page.build()]
        self.page.update()

    async def add_new_rule(self, e):
        if len(self.station.rules) >= 10:
            self.page.open(
                ft.SnackBar(content=ft.Text(value='Нельзя создать больше 10 правил'), bgcolor=ft.Colors.ORANGE))
        else:
            self.rule_option = ft.Dropdown(width=125, options=[
                ft.DropdownOption('равна'),
                ft.DropdownOption('больше'),
                ft.DropdownOption('меньше'),
            ])
            self.rule_sign = ft.Dropdown(width=73, options=[ft.DropdownOption('+'), ft.DropdownOption('-')])
            self.rule_value = ft.TextField(input_filter=ft.NumbersOnlyInputFilter(), width=50)
            self.rule_period = ft.TextField(input_filter=ft.NumbersOnlyInputFilter(), width=50)
            self.page.overlay.append(ft.Container(bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
                                                  padding=20, expand=True,
                                                  alignment=ft.alignment.center,
                                                  content=ft.Container(content=ft.Column(controls=[
                                                      ft.Container(content=ft.Text(value='Создать новое правило',
                                                                                   color=ft.Colors.BLACK, expand=True,
                                                                                   size=30,
                                                                                   weight=ft.FontWeight.BOLD,
                                                                                   text_align=ft.TextAlign.CENTER),
                                                                   expand=True,
                                                                   height=50,
                                                                   alignment=ft.alignment.top_center),
                                                      ft.Row(controls=[
                                                          ft.Text(value='Температура', color=ft.Colors.BLACK),
                                                          self.rule_option,
                                                          self.rule_sign,
                                                          self.rule_value,
                                                          ft.Text(value='через', color=ft.Colors.BLACK),
                                                          self.rule_period,
                                                          ft.Text(value='часа/часов', color=ft.Colors.BLACK),
                                                      ], expand=True),
                                                      ft.Row(controls=[
                                                          ft.ElevatedButton("✅ Создать", on_click=self.add_rule),
                                                          ft.ElevatedButton("❌ Отменить", on_click=self.close_overlay)
                                                      ], alignment=ft.alignment.bottom_center)
                                                  ], height=250, width=600, alignment=ft.alignment.center),
                                                      height=250,
                                                      width=600, bgcolor=ft.Colors.WHITE,
                                                      border_radius=20, padding=20, expand=True)))
            self.page.update()

    def close_overlay(self, e):
        self.page.overlay.clear()
        self.page.update()

    async def add_rule(self, e):
        if self.app_manager.is_requesting:
            pass
        else:
            if self.rule_period.value > 168:
                self.page.open(
                    ft.SnackBar(content=ft.Text(value='Период не может быть больше 168 часов (неделя)'), bgcolor=ft.Colors.ORANGE))
            else:
                self.close_overlay(None)
                async with ClientSession() as session:
                    try:
                        self.app_manager.is_requesting = True
                        await session.post(
                            ADD_RULE_URL + f'?station_id={self.station.id}&rule_option={self.rule_option.value}&rule_value={self.rule_sign.value}{self.rule_value.value}&rule_period={self.rule_period.value}'
                        )
                    except Exception as e:
                        self.page.open(
                            ft.SnackBar(content=ft.Text(value='Сервер недоступен, запрос отклонен'), bgcolor=ft.Colors.RED))
                        print(f"Error: {e}")
                    finally:
                        await session.close()
                        self.app_manager.is_requesting = False

    async def delete_rule(self, rule_id):
        if self.app_manager.is_requesting:
            pass
        else:
            async with ClientSession() as session:
                try:
                    self.app_manager.is_requesting = True
                    await session.post(
                        DELETE_RULE_URL + f'?station_id={self.station.id}&rule_id={rule_id}'
                    )
                except Exception as e:
                    self.page.open(
                        ft.SnackBar(content=ft.Text(value='Сервер недоступен, запрос отклонен'), bgcolor=ft.Colors.RED))
                    print(f"Error: {e}")
                finally:
                    await session.close()
                    self.app_manager.is_requesting = False

    async def toggle_station(self, e):
        if self.app_manager.is_requesting:
            pass
        else:
            async with ClientSession() as session:
                try:
                    self.app_manager.is_requesting = True
                    await session.post(
                        TOGGLE_URL + f'?station_id={self.station.id}'
                    )
                except Exception as e:
                    self.page.open(
                        ft.SnackBar(content=ft.Text(value='Сервер недоступен, запрос отклонен'), bgcolor=ft.Colors.RED))
                    print(f"Error: {e}")
                finally:
                    await session.close()
                    self.app_manager.is_requesting = False

    async def station_delete(self, e):
        if self.app_manager.is_requesting:
            pass
        else:
            async with ClientSession() as session:
                try:
                    self.app_manager.is_requesting = True
                    await session.post(
                        DELETE_URL + f'?station_id={self.station.id}'
                    )
                except Exception as e:
                    self.page.open(
                        ft.SnackBar(content=ft.Text(value='Сервер недоступен, запрос отклонен'), bgcolor=ft.Colors.RED))
                    print(f"Error: {e}")
                finally:
                    await session.close()
                    self.app_manager.is_requesting = False

    async def station_reload(self, e):
        if self.app_manager.is_requesting:
            pass
        else:
            async with ClientSession() as session:
                try:
                    self.app_manager.is_requesting = True
                    await session.post(
                        RELOAD_URL + f'?station_id={self.station.id}'
                    )
                except Exception as e:
                    self.page.open(
                        ft.SnackBar(content=ft.Text(value='Сервер недоступен, запрос отклонен'), bgcolor=ft.Colors.RED))
                    print(f"Error: {e}")
                finally:
                    await session.close()
                    self.app_manager.is_requesting = False

    async def edit_station(self, e):
        self.station_newname = ft.TextField(hint_text='Новое название')
        self.station_newdesc = ft.TextField(hint_text='Новое описание')
        self.page.overlay.append(ft.Container(visible=True, bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
                                              padding=20, expand=True,
                                              alignment=ft.alignment.center,
                                              content=ft.Container(content=ft.Column(controls=[
                                                  ft.Container(content=ft.Text(value='Редактировать станцию',
                                                                               color=ft.Colors.BLACK, expand=True,
                                                                               size=30,
                                                                               weight=ft.FontWeight.BOLD,
                                                                               text_align=ft.TextAlign.CENTER),
                                                               expand=True,
                                                               height=50,
                                                               alignment=ft.alignment.top_center),
                                                  ft.Column(controls=[
                                                      self.station_newname,
                                                      self.station_newdesc
                                                  ]),
                                                  ft.Row(controls=[
                                                      ft.ElevatedButton("✅ Применить", on_click=self.edit_stattion),
                                                      ft.ElevatedButton("❌ Отменить", on_click=self.close_overlay)
                                                  ], alignment=ft.alignment.bottom_center)
                                              ], height=250, width=600, alignment=ft.alignment.center), height=250,
                                                  width=600, bgcolor=ft.Colors.WHITE,
                                                  border_radius=20, padding=20, expand=True)))

        self.page.update()

    async def edit_stattion(self, e):
        if self.app_manager.is_requesting:
            pass
        else:
            self.close_overlay(None)
            async with ClientSession() as session:
                try:
                    self.app_manager.is_requesting = True
                    await session.post(
                        EDIT_STATION_URL + f'?station_id={self.station.id}&name={self.station_newname.value}&descr={self.station_newdesc.value}'
                    )
                except Exception as e:
                    self.page.open(
                        ft.SnackBar(content=ft.Text(value='Сервер недоступен, запрос отклонен'), bgcolor=ft.Colors.RED))
                    print(f"Error: {e}")
                finally:
                    await session.close()
                    self.app_manager.is_requesting = False


class MainPage:
    def __init__(self, stations, page, logo, app):
        self.stations_list_controls = None
        self.stations_list = None
        self.station_list = None
        self.marker_layer = None
        self.map = None
        self.stations = stations
        self.page = page
        self.logo = logo
        self.tile_layer = None
        self.attr_layer = None
        self.last_click_time = None
        self.app_manager = app

    def build(self):
        self.tile_layer = map.TileLayer(
            url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            tile_bounds=map.MapLatitudeLongitudeBounds(map.MapLatitudeLongitude(85.0, -180.0),
                                                       map.MapLatitudeLongitude(-85.0, 180.0))
        )

        self.attr_layer = map.RichAttribution(
            attributions=[
                map.TextSourceAttribution(
                    text="OpenStreetMap Contributors",
                    on_click=lambda e: e.page.launch_url(
                        "https://openstreetmap.org/copyright"
                    ),
                ),
            ],
            alignment=ft.alignment.bottom_right,
            show_flutter_map_attribution=False
        )

        self.marker_layer = map.MarkerLayer(markers=[
            self._create_marker_for_station(station) for station in self.stations.values()
        ])

        # Создание карты
        self.map = map.Map(
            initial_center=map.MapLatitudeLongitude(55.7558, 37.6176),  # Москва
            initial_zoom=3,
            min_zoom=3,
            max_zoom=12,
            on_secondary_tap=self.add_new_station_coords,
            layers=[self.tile_layer, self.attr_layer, self.marker_layer],
            interaction_configuration=map.MapInteractionConfiguration(
                flags=map.MapInteractiveFlag.SCROLL_WHEEL_ZOOM | map.MapInteractiveFlag.DRAG
            ), expand=True
        )

        self.stations_list_controls = ft.Column(scroll=ft.ScrollMode.ALWAYS, controls=[
            ft.Container(content=ft.Row(expand=True, height=20, controls=[
                ft.Container(content=self._get_station_icon(station), width=40),
                ft.Container(content=ft.Text(value=station.name,
                                             color=ft.Colors.BLACK, width=190, height=20,
                                             text_align=ft.TextAlign.CENTER, no_wrap=True), height=20,
                             border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2)),
                             width=190),
                ft.Container(content=ft.Text(value=f'({station.lat}, {station.lng})', color=ft.Colors.BLUE,
                                             text_align=ft.TextAlign.CENTER, no_wrap=True),
                             on_click=lambda e, lat=station.lat, lng=station.lng: self._on_coord_click(lat, lng),
                             width=140, height=20),
                ft.Container(content=ft.Text(
                    value=station.description,
                    color=ft.Colors.BLACK, height=20, width=480, text_align=ft.TextAlign.CENTER, no_wrap=True),
                    height=20, border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2)),
                    width=480),
                ft.Text(value=f'{station.updated_at}' if station.updated_at else 'Никогда', color=ft.Colors.BLACK,
                        width=180, height=20, text_align=ft.TextAlign.CENTER, no_wrap=True),
                ft.Container(
                    content=ft.Text(value=f'{station.created_at}', color=ft.Colors.BLACK, width=180, height=20,
                                    text_align=ft.TextAlign.CENTER, no_wrap=True), height=20,
                    border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2)), width=180),
                ft.ElevatedButton(text='Перейти', icon=ft.Icons.ARROW_RIGHT,
                                  on_click=lambda e, s=station: self.to_station(s), height=20)
            ]), key=str(station.id), bgcolor=ft.Colors.GREY_100) for station in self.stations.values()
        ], expand=True)

        self.stations_list = ft.Column(controls=[
            ft.Row(controls=[
                ft.Row(controls=[
                    ft.Text(value=f'Всего: {len(self.stations)}', color=ft.Colors.BLACK),
                    ft.Text(
                        value=f'Активных: {sum(1 for station in self.stations.values() if station.status == "В норме")}',
                        color=ft.Colors.BLACK),
                    ft.Text(
                        value=f'Ошибки: {sum(1 for station in self.stations.values() if station.status == "Ошибка")}',
                        color=ft.Colors.BLACK),
                ]),
                ft.ElevatedButton(text='+ Добавить станцию', color=ft.Colors.BLACK, on_click=self.add_new_station,
                                  disabled=len(self.stations) >= 25)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Container(content=ft.Row(height=20, controls=[
                ft.Container(height=20, width=40),
                ft.Container(content=ft.Text(value='Название', height=20, text_align=ft.TextAlign.CENTER), height=20,
                             width=190,
                             border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2))),
                ft.Container(content=ft.Text(value='Координаты', height=20, text_align=ft.TextAlign.CENTER), height=20,
                             width=140),
                ft.Container(content=ft.Text(value='Описание', height=20, text_align=ft.TextAlign.CENTER), height=20,
                             width=480,
                             border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2))),
                ft.Container(content=ft.Text(value='Обновлено', height=20, text_align=ft.TextAlign.CENTER), height=20,
                             width=180),
                ft.Container(content=ft.Text(value='Добавлено', height=20, text_align=ft.TextAlign.CENTER), height=20,
                             width=180,
                             border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2)))
            ]), border=ft.Border(bottom=ft.BorderSide(width=2, color=ft.Colors.BLACK),
                                 top=ft.BorderSide(width=2, color=ft.Colors.BLACK))),
            self.stations_list_controls
        ], height=150)

        return ft.Column(controls=[
            ft.Row(
                controls=[
                    ft.Image(src_base64=self.logo, fit=ft.ImageFit.CONTAIN, width=60,
                             height=60),
                    ft.Text('Feyra weather', color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD, size=40),
                ]
            ),
            ft.Container(content=self.map, expand=True),
            ft.Container(content=self.stations_list, height=300, border_radius=15,
                         border=ft.border.all(color=ft.Colors.BLUE, width=1), padding=8)
        ], expand=True)

    def update(self):

        self.marker_layer.markers = [
            self._create_marker_for_station(station) for station in self.stations.values()
        ]
        self.map.layers = [self.tile_layer, self.attr_layer, self.marker_layer]

        self.stations_list_controls.controls = [
            ft.Container(content=ft.Row(expand=True, height=20, controls=[
                ft.Container(content=self._get_station_icon(station), width=40),
                ft.Container(content=ft.Text(value=station.name,
                                             color=ft.Colors.BLACK, width=190, height=20,
                                             text_align=ft.TextAlign.CENTER, no_wrap=True), height=20,
                             border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2)),
                             width=190),
                ft.Container(content=ft.Text(value=f'({station.lat}, {station.lng})', color=ft.Colors.BLUE,
                                             text_align=ft.TextAlign.CENTER, no_wrap=True),
                             on_click=lambda e, lat=station.lat, lng=station.lng: self._on_coord_click(lat, lng),
                             width=140, height=20),
                ft.Container(content=ft.Text(
                    value=station.description,
                    color=ft.Colors.BLACK, height=20, width=480, text_align=ft.TextAlign.CENTER, no_wrap=True),
                    height=20, border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2)),
                    width=480),
                ft.Text(value=f'{station.updated_at}' if station.updated_at else 'Никогда', color=ft.Colors.BLACK,
                        width=180, height=20, text_align=ft.TextAlign.CENTER, no_wrap=True),
                ft.Container(
                    content=ft.Text(value=f'{station.created_at}', color=ft.Colors.BLACK, width=180, height=20,
                                    text_align=ft.TextAlign.CENTER, no_wrap=True), height=20,
                    border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2)), width=180),
                ft.ElevatedButton(text='Перейти', icon=ft.Icons.ARROW_RIGHT,
                                  on_click=lambda e, s=station: self.to_station(s), height=20)
            ]), key=str(station.id), bgcolor=ft.Colors.GREY_100) for station in self.stations.values()
        ]

        self.stations_list.controls = [
            ft.Row(controls=[
                ft.Row(controls=[
                    ft.Text(value=f'Всего: {len(self.stations)}', color=ft.Colors.BLACK),
                    ft.Text(
                        value=f'Активных: {sum(1 for station in self.stations.values() if station.status == "В норме")}',
                        color=ft.Colors.BLACK),
                    ft.Text(
                        value=f'Ошибки: {sum(1 for station in self.stations.values() if station.status == "Ошибка")}',
                        color=ft.Colors.BLACK),
                ]),
                ft.ElevatedButton(text='+ Добавить станцию', color=ft.Colors.BLACK, on_click=self.add_new_station,
                                  disabled=len(self.stations) >= 25)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Container(content=ft.Row(height=20, controls=[
                ft.Container(height=20, width=40),
                ft.Container(content=ft.Text(value='Название', height=20, text_align=ft.TextAlign.CENTER), height=20,
                             width=190,
                             border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2))),
                ft.Container(content=ft.Text(value='Координаты', height=20, text_align=ft.TextAlign.CENTER), height=20,
                             width=140),
                ft.Container(content=ft.Text(value='Описание', height=20, text_align=ft.TextAlign.CENTER), height=20,
                             width=480,
                             border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2))),
                ft.Container(content=ft.Text(value='Обновлено', height=20, text_align=ft.TextAlign.CENTER), height=20,
                             width=180),
                ft.Container(content=ft.Text(value='Добавлено', height=20, text_align=ft.TextAlign.CENTER), height=20,
                             width=180,
                             border=ft.border.symmetric(horizontal=ft.BorderSide(color=ft.Colors.BLACK, width=2)))
            ]), border=ft.Border(bottom=ft.BorderSide(width=2, color=ft.Colors.BLACK),
                                 top=ft.BorderSide(width=2, color=ft.Colors.BLACK))),
            self.stations_list_controls
        ]

        self.page.update()

    def _on_coord_click(self, lat, lng):
        now = datetime.datetime.now()
        if self.last_click_time and (now - self.last_click_time).total_seconds() < 1:
            self.page.open(ft.SnackBar(content=ft.Text(value='Не так быстро!'), bgcolor=ft.Colors.ORANGE))
        self.last_click_time = now
        self.map.center_on(zoom=8, point=map.MapLatitudeLongitude(lat, lng), animation_duration=0)

    def _get_station_icon(self, station: Station, is_marker=False):
        if station.status == 'В норме':
            icon = ft.Icons.CHECK_CIRCLE
            icon_color = ft.Colors.GREEN
        elif station.status == 'Обновление':
            icon = ft.Icons.UPDATE
            icon_color = ft.Colors.YELLOW_800
        elif station.status == 'Выключено':
            icon = ft.Icons.OFFLINE_PIN
            icon_color = ft.Colors.GREY
        else:
            icon = ft.Icons.ERROR
            icon_color = ft.Colors.RED

        if station.alerts and station.status == 'В норме':
            if is_marker:
                status_icon = ft.Text(value='🚨', color=ft.Colors.BLACK,
                                      text_align=ft.TextAlign.CENTER, width=40, height=40, size=15)
            else:
                status_icon = ft.Text(value=str(len(station.alerts)) + ' 🚨', color=ft.Colors.BLACK,
                                      text_align=ft.TextAlign.CENTER)
        else:
            status_icon = ft.Container(content=ft.Icon(
                name=icon,
                color=icon_color,
                size=20,
            ))

        return status_icon

    def _create_marker_for_station(self, station: Station) -> map.Marker:
        """Создание маркера для станции на карте"""
        # Цвет маркера в зависимости от статуса
        if station.status == 'В норме':
            marker_color = ft.Colors.GREEN
        elif station.status == 'Обновление':
            marker_color = ft.Colors.YELLOW_300
        elif station.status == 'Выключено':
            marker_color = ft.Colors.GREY
        else:
            marker_color = ft.Colors.RED

        status_icon = self._get_station_icon(station, is_marker=True)

        # Контент маркера
        marker_content = ft.Container(
            content=status_icon,
            width=30,
            height=30,
            bgcolor=ft.Colors.WHITE,
            border_radius=15,
            border=ft.border.all(2, marker_color),
            on_click=lambda e: self.scroll_to_key(station.id),
            on_long_press=lambda e: self.to_station(station),
            tooltip=ft.Tooltip(message=station.name[:15], text_style=ft.TextStyle(color=ft.Colors.BLACK, size=12),
                               bgcolor=ft.Colors.LIGHT_BLUE_50, border=ft.border.all(width=1, color=ft.Colors.GREY_300))
        )

        return map.Marker(
            content=marker_content,
            coordinates=map.MapLatitudeLongitude(station.lat, station.lng), width=30, height=30
        )

    def scroll_to_key(self, st_id):
        key = str(st_id)
        self.stations_list_controls.scroll_to(key=key, duration=500)
        for control in self.stations_list_controls.controls:
            if control.key == key:
                asyncio.run(self.color_scroll(control))
                break

    async def color_scroll(self, control):
        # Find the container in the column
        control.bgcolor = ft.Colors.BLUE_GREY_300
        control.update()

        await asyncio.sleep(1)

        # Restore original color
        control.bgcolor = ft.Colors.GREY_100
        control.update()

    def to_station(self, station: Station):
        asyncio.run(self.to_station_page(station))

    async def to_station_page(self, station: Station):
        station_page = StationPage(page=self.page, station=station, logo=self.logo, app=self.app_manager)
        self.app_manager.current_page = station_page
        self.page.controls = [station_page.build()]
        self.page.update()

    async def add_new_station_coords(self, e: map.MapTapEvent):
        await self.add_new_station(e=None, lat=e.coordinates.latitude, lng=e.coordinates.longitude)

    async def add_new_station(self, e, lat=None, lng=None):
        station_newname = ft.TextField(hint_text='Название')
        station_newdesc = ft.TextField(hint_text='Описание')
        station_lat = ft.TextField(hint_text='Широта', value=lat, width=275)
        station_lng = ft.TextField(hint_text='Долгота', value=lng, width=275)
        activate_checkbox = ft.Checkbox(label='Активировать станцию?', value=False)

        def clicked(e):
            self.new_stattion(None, station_newname.value, station_newdesc.value, station_lat.value, station_lng.value, activate_checkbox.value)

        self.page.overlay.append(ft.Container(visible=True, bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
                                              padding=20, expand=True,
                                              alignment=ft.alignment.center,
                                              content=ft.Container(content=ft.Column(controls=[
                                                  ft.Container(content=ft.Text(value='Создать станцию',
                                                                               color=ft.Colors.BLACK, expand=True,
                                                                               size=30,
                                                                               weight=ft.FontWeight.BOLD,
                                                                               text_align=ft.TextAlign.CENTER),
                                                               expand=True,
                                                               height=50,
                                                               alignment=ft.alignment.top_center),
                                                  ft.Column(controls=[
                                                      station_newname,
                                                      station_newdesc,
                                                      ft.Row(controls=[
                                                          station_lat,
                                                          station_lng
                                                      ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                                                      activate_checkbox
                                                  ]),
                                                  ft.Row(controls=[
                                                      ft.ElevatedButton("✅ Создать",
                                                                        on_click=clicked),
                                                      ft.ElevatedButton("❌ Отменить", on_click=self.close_overlay)
                                                  ], alignment=ft.alignment.bottom_center)
                                              ], height=300, width=600, alignment=ft.alignment.center), height=300,
                                                  width=600, bgcolor=ft.Colors.WHITE,
                                                  border_radius=20, padding=20, expand=True)))


        self.page.update()

    def new_stattion(self, e, name, descr, lat, lng, activate):
        self.close_overlay(None)
        asyncio.run(self.new_station(None, name, descr, lat, lng, activate))

    async def new_station(self, e, name, descr, lat, lng, activate):
        if self.app_manager.is_requesting:
            pass
        else:
            async with ClientSession() as session:
                try:
                    self.app_manager.is_requesting = True
                    await session.post(
                        ADD_STATION_URL + f'?name={name}&descr={descr}&lat={lat}&lng={lng}&activate={activate}'
                    )
                except Exception as e:
                    self.page.open(
                        ft.SnackBar(content=ft.Text(value='Сервер недоступен, запрос отклонен'), bgcolor=ft.Colors.RED))
                    print(f"Error: {e}")
                finally:
                    await session.close()
                    self.app_manager.is_requesting = False

    def close_overlay(self, e):
        self.page.overlay.clear()
        self.page.update()
