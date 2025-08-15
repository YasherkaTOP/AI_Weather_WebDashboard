import logging
import datetime
import pickle
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd

from scipy.interpolate import PchipInterpolator
from statsmodels.tsa.seasonal import STL

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer

from config import OPEN_METEO_ARCHIVE_URL, OPEN_METEO_FORECAST_URL, MODEL_PATH, PARAMS_PATH, ENCODER_PATH

warnings.filterwarnings('ignore')

CONFIG_PATH = 'config.py'

# Используем логгирование
logger = logging.getLogger("forecast_task")
logging.basicConfig(level=logging.INFO)

try:
    model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
    with open(PARAMS_PATH, 'rb') as f:
        params = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
except Exception as e:
    logger.exception("Failed to load config %s", CONFIG_PATH, e)
    raise


def _validate_lat_lng(lat: float, lng: float) -> None:
    """Валидация координат"""
    if not isinstance(lat, (float, int)) or not isinstance(lng, (float, int)):
        raise ValueError("lat and lng must be numeric")
    lat = float(lat)
    lng = float(lng)
    if not (-90.0 <= lat <= 90.0):
        raise ValueError("lat out of bounds [-90,90]")
    if not (-180.0 <= lng <= 180.0):
        raise ValueError("lng out of bounds [-180,180]")


def process_task(lat, lng) -> Tuple[List[float], List[datetime.datetime]]:
    """Public entry point used by FastAPI background task"""
    logger.info("Starting process_task for lat=%s lng=%s", lat, lng)
    try:
        df = get_data(lat, lng)
        preds_df = get_prediction(df)
        logger.info("process_task succeeded for lat=%s lng=%s", lat, lng)
        return preds_df['temp'].tolist(), preds_df['time'].tolist()
    except Exception as e:
        logger.exception("process_task failed for lat=%s lng=%s: %s", lat, lng, e)
        raise


# ПРИ ИЗМЕНЕНИИ АЛГОРИТМА ПРЕДСКАЗАНИЯ ИЛИ ЕГО ПАРАМЕТРОВ ПОДГОТОВКИ ДАННЫХ ДАННАЯ ФУНКЦИЯ ПЕРЕСТАЕТ РАБОТАТЬ ПРАВИЛЬНО
def correct_adjustment(test_df, predicted_values, max_deviation=2, transition_points=None):
    """Корректировка предсказаний с использованием кубической интерполяции Эрмита."""
    true_value_at_minus1 = test_df['temp'].iloc[-745]
    n = len(predicted_values)

    if transition_points is None:
        # По умолчанию используем переходный участок в 20% от длины данных
        transition_points = [0, int(n * 0.1), int(n * 0.2)]

    # Вычисляем допустимые границы для первого предсказанного значения
    min_acceptable = true_value_at_minus1 - max_deviation
    max_acceptable = true_value_at_minus1 + max_deviation

    # Вычисляем необходимую корректировку для первого значения
    if predicted_values[0] < min_acceptable:
        initial_correction = min_acceptable - predicted_values[0]
    elif predicted_values[0] > max_acceptable:
        initial_correction = max_acceptable - predicted_values[0]
    else:
        initial_correction = 0  # Значение уже находится в допустимом диапазоне

    # Создаём точки для интерполяции
    x_points = [-1] + transition_points + [n - 1]

    # Значения корректировки в этих точках (плавное убывание)
    y_corrections = [initial_correction, initial_correction, initial_correction * 0.3, 0, 0]

    # Создаём интерполятор
    interpolator = PchipInterpolator(x_points, y_corrections)

    # Создаём массив индексов для всех точек
    all_indices = np.arange(-1, n)

    # Вычисляем корректировки для всех точек
    all_corrections = interpolator(all_indices)

    # Применяем корректировки только к прогнозам (индексы от 0 до n-1)
    corrections = all_corrections[1:]

    # Применяем корректировки к прогнозам
    adjusted_values = predicted_values + corrections

    return adjusted_values, corrections


# ПРИ ИЗМЕНЕНИИ АЛГОРИТМА ПРЕДСКАЗАНИЯ ИЛИ ЕГО ПАРАМЕТРОВ ПОДГОТОВКИ ДАННЫХ ДАННАЯ ФУНКЦИЯ ПЕРЕСТАЕТ РАБОТАТЬ ПРАВИЛЬНО
def get_data(lat, lng):
    _validate_lat_lng(lat, lng)

    start_date = (datetime.datetime.utcnow() - datetime.timedelta(days=61)).strftime('%Y-%m-%d')
    end_date = (datetime.datetime.utcnow() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')

    try:
        import requests_cache
        from retry_requests import retry
        import openmeteo_requests
    except Exception:
        logger.exception("Missing optional dependencies for Open-Meteo client — ensure they're installed")
        raise

    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params_archive = {
        "latitude": float(lat),
        "longitude": float(lng),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m"],
        "timezone": "GMT",
    }

    try:
        responses = openmeteo.weather_api(OPEN_METEO_ARCHIVE_URL, params=params_archive)
        hourly = responses[0].Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    except Exception:
        logger.exception("Failed to fetch archive data from Open-Meteo")
        raise

    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "temp": hourly_temperature_2m,
    }
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    params_forecast = {
        "latitude": float(lat),
        "longitude": float(lng),
        "hourly": "temperature_2m",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "GMT"
    }
    try:
        responses1 = openmeteo.weather_api(OPEN_METEO_FORECAST_URL, params=params_forecast)
        hourly1 = responses1[0].Hourly()
        hourly_temperature_2m1 = hourly1.Variables(0).ValuesAsNumpy()
    except Exception:
        logger.exception("Failed to fetch forecast data from Open-Meteo")
        raise

    hourly_data1 = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly1.Time(), unit="s"),
            end=pd.to_datetime(hourly1.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly1.Interval()),
            inclusive="left",
        ),
        "temp": hourly_temperature_2m1,
    }
    hourly_dataframe1 = pd.DataFrame(data=hourly_data1)

    data = pd.concat([hourly_dataframe, hourly_dataframe1], ignore_index=True)

    # Базовые признаки
    data['latitude'] = lat
    data['longitude'] = lng
    data['time'] = pd.to_datetime(data['time'])

    # Временные признаки
    data['hour'] = data['time'].dt.hour
    data['month'] = data['time'].dt.month
    data['season'] = (data['month'] % 12 // 3).astype(int)

    # Циклические признаки
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    # Лаги
    lags = [1, 24]
    for lag in lags:
        data[f'temp_lag_{lag}'] = data['temp'].shift(lag)

    # Скользящее окно
    windows = [168, 672]
    for window in windows:
        # Среднее
        data[f'temp_rolling_mean_{window}'] = data['temp'].rolling(window=window, min_periods=1).mean()
        # Стандартное отклонение
        data[f'temp_rolling_std_{window}'] = data['temp'].rolling(window=window, min_periods=1).std()
        # Минимум
        data[f'temp_rolling_min_{window}'] = data['temp'].rolling(window=window, min_periods=1).min()
        # Максимум
        data[f'temp_rolling_max_{window}'] = data['temp'].rolling(window=window, min_periods=1).max()

    # Разница
    diffs = [1, 24]
    for diff in diffs:
        data[f'temp_diff_{diff}'] = data['temp'].diff(diff)
        # Процентное изменение
        data[f'temp_pct_change_{diff}'] = data['temp'].pct_change(diff)

    # STL-декомпозиция
    try:
        stl = STL(data['temp'].fillna(method='ffill'), period=24, robust=True)
        result = stl.fit()

        data[f'temp_trend'] = result.trend.reindex(data.index).fillna(method='ffill').fillna(0)
        data[f'temp_seasonal'] = result.seasonal.reindex(data.index).fillna(0)
        residual = result.resid

        # Определение порогов для аномалий
        residual_mean = residual.mean()
        residual_std = residual.std()

        # Добавление абсолютного значения отклонения (для ранжирования аномалий)
        data[f'temp_anomaly_score'] = np.abs((residual - residual_mean) / (residual_std + 1e-10))

    except:
        data[f'temp_trend'] = 0
        data[f'temp_seasonal'] = 0
        data[f'temp_anomaly_score'] = 0

    data['time_idx'] = range(0, len(data))

    data.dropna(inplace=True)

    # Формирование данных на горизонт предсказаний
    last_index = data['time_idx'].max()
    new_df = pd.DataFrame(
        {'time': pd.date_range(start=pd.to_datetime(data['time'].iloc[-1]) + datetime.timedelta(hours=1),
                               periods=672, freq='H')})
    new_df['hour'] = new_df['time'].dt.hour
    new_df['month'] = new_df['time'].dt.month
    new_df['season'] = (new_df['month'] % 12 // 3).astype(int)

    new_df['hour_sin'] = np.sin(2 * np.pi * new_df['hour'] / 24)
    new_df['hour_cos'] = np.cos(2 * np.pi * new_df['hour'] / 24)
    new_df['month_sin'] = np.sin(2 * np.pi * new_df['month'] / 12)
    new_df['month_cos'] = np.cos(2 * np.pi * new_df['month'] / 12)

    new_df['time_idx'] = range(last_index + 1, last_index + 1 + len(new_df))
    new_df.fillna(0, inplace=True)

    df = pd.concat([data, new_df], ignore_index=True)
    df['station_id'] = 999
    df.reset_index(drop=True, inplace=True)

    return df


# ПРИ ИЗМЕНЕНИИ АЛГОРИТМА ПРЕДСКАЗАНИЯ ИЛИ ЕГО ПАРАМЕТРОВ ПОДГОТОВКИ ДАННЫХ ДАННАЯ ФУНКЦИЯ ПЕРЕСТАЕТ РАБОТАТЬ ПРАВИЛЬНО
def get_prediction(data):
    """Подготовка набора данных, запуск модели. Возвращает прогнозируемый фрейм данных"""

    # Создание TimeSeriesDataSet
    try:
        dataset = TimeSeriesDataSet.from_parameters(parameters=params, data=data, predict=True, stop_randomization=True, categorical_encoders={"series": encoder})
    except Exception:
        logger.exception("Failed to construct TimeSeriesDataSet")
        raise

    dataloader = dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    try:
        preds = model.predict(dataloader)
    except Exception:
        logger.exception("Model prediction failed")
        raise

    point_predictions = preds.cpu().numpy().squeeze()

    predicted_values, _ = correct_adjustment(data, point_predictions)

    dates = data.iloc[-744:]['time'].values
    predicted_data = pd.DataFrame({'time': dates, 'temp': predicted_values})
    predicted_data = pd.concat([data[['time', 'temp']].iloc[-745:-744], predicted_data.iloc[:672]], ignore_index=True)
    predicted_data['time'] = pd.to_datetime(predicted_data['time'])
    return predicted_data
