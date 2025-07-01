import pandas as pd
import numpy as np
import pickle
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer
import os
import requests_cache
from retry_requests import retry
import openmeteo_requests
import datetime
import warnings

from scipy.interpolate import PchipInterpolator
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')


def process_task(lat, lng):
    try:
        preds = get_prediction(data=get_data(lat, lng), dataset_path='dataset_parameters.pkl', model_path='model.ckpt')
        return preds['temp'].tolist(), preds['time'].tolist()
    except:
        return 'error'


def correct_adjustment(test_df, predicted_values, max_deviation=2, transition_points=None):
    """
    ИСПРАВЛЕННАЯ версия корректировки с использованием кубической интерполяции Эрмита.

    Args:
        true_value_at_minus1: Значение на -1 шаге
        predicted_values: Массив предсказанных значений
        max_deviation: Максимально допустимое отклонение от значения на -1 шаге
        transition_points: Список точек для определения переходного участка
    """
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


def create_feature_interaction(df, feature1, feature2, operation='multiply'):
    """
    Создание взаимодействий между признаками

    Parameters:
    -----------
    df : pd.DataFrame
        Датафрейм с признаками
    feature1 : str
        Первый признак для взаимодействия
    feature2 : str
        Второй признак для взаимодействия
    operation : str, default='multiply'
        Операция взаимодействия: 'multiply', 'add', 'subtract', 'divide'

    Returns:
    --------
    pd.DataFrame
        Датафрейм с добавленным взаимодействием признаков
    """
    df_interaction = df.copy()

    # Проверка наличия признаков
    if feature1 not in df.columns or feature2 not in df.columns:
        return df_interaction

    # Создание взаимодействия
    if operation == 'multiply':
        df_interaction[f'{feature1}_x_{feature2}'] = df_interaction[feature1] * df_interaction[feature2]
    elif operation == 'add':
        df_interaction[f'{feature1}_+_{feature2}'] = df_interaction[feature1] + df_interaction[feature2]
    elif operation == 'subtract':
        df_interaction[f'{feature1}_-_{feature2}'] = df_interaction[feature1] - df_interaction[feature2]
    elif operation == 'divide':
        # Защита от деления на 0
        df_interaction[f'{feature1}_/_{feature2}'] = df_interaction[feature1] / (df_interaction[feature2] + 1e-10)

    return df_interaction


def get_data(lat, lng):
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    start_date = (datetime.datetime.now() - datetime.timedelta(days=105)).strftime('%Y-%m-%d')
    end_date = (datetime.datetime.utcnow() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": f"{start_date}",
        "end_date": f"{end_date}",
        "hourly": ["temperature_2m"],
        "timezone": "GMT"
    }
    responses = openmeteo.weather_api(url, params=params)

    hourly = responses[0].Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = {"time": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ), "temp": hourly_temperature_2m}

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe = hourly_dataframe[
        hourly_dataframe['time'] >= pd.to_datetime(datetime.datetime.utcnow() - datetime.timedelta(hours=2520))]

    url1 = "https://api.open-meteo.com/v1/forecast"
    params1 = {
        "latitude": lat,
        "longitude": lng,
        "hourly": "temperature_2m",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "GMT"
    }
    responses1 = openmeteo.weather_api(url1, params=params1)

    hourly1 = responses1[0].Hourly()
    hourly_temperature_2m1 = hourly1.Variables(0).ValuesAsNumpy()

    hourly_data1 = {"time": pd.date_range(
        start=pd.to_datetime(hourly1.Time(), unit="s"),
        end=pd.to_datetime(hourly1.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly1.Interval()),
        inclusive="left"
    ), "temp": hourly_temperature_2m1}

    hourly_dataframe1 = pd.DataFrame(data=hourly_data1)
    hourly_dataframe1 = hourly_dataframe1[hourly_dataframe1['time'] <= pd.to_datetime(datetime.datetime.utcnow())]

    data = pd.concat([hourly_dataframe, hourly_dataframe1])
    data = data.reset_index()
    data.drop(['index'], axis=1, inplace=True)

    data['latitude'] = lat
    data['longitude'] = lng
    data['station_id'] = 1
    data['time'] = pd.to_datetime(data['time'])

    data['hour'] = data['time'].dt.hour
    data['day'] = data['time'].dt.day
    data['month'] = data['time'].dt.month
    data['year'] = data['time'].dt.year
    data['dayofweek'] = data['time'].dt.dayofweek
    data['quarter'] = data['time'].dt.quarter
    data['dayofyear'] = data['time'].dt.dayofyear
    data['weekofyear'] = data['time'].dt.isocalendar().week

    # День или ночь (приблизительно)
    data['is_day'] = ((data['hour'] >= 6) & (data['hour'] <= 18)).astype(int)

    # Выходной день или нет
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)

    # Сезоны (северное полушарие): 0-зима, 1-весна, 2-лето, 3-осень
    data['season'] = (data['month'] % 12 // 3).astype(int)

    # Циклические признаки для периодических переменных
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)

    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
    data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)

    data['season_sin'] = np.sin(2 * np.pi * data['season'] / 4)
    data['season_cos'] = np.cos(2 * np.pi * data['season'] / 4)

    lags = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720, 1080]
    for lag in lags:
        data[f'temp_lag_{lag}'] = data['temp'].shift(lag)

    windows = [3, 6, 12, 24, 48, 72, 168, 336, 720, 1080]
    for window in windows:
        # Среднее
        data[f'temp_rolling_mean_{window}'] = data['temp'].rolling(window=window, min_periods=1).mean()
        # Стандартное отклонение
        data[f'temp_rolling_std_{window}'] = data['temp'].rolling(window=window, min_periods=1).std()
        # Минимум
        data[f'temp_rolling_min_{window}'] = data['temp'].rolling(window=window, min_periods=1).min()
        # Максимум
        data[f'temp_rolling_max_{window}'] = data['temp'].rolling(window=window, min_periods=1).max()
        # Размах
        data[f'temp_rolling_range_{window}'] = (
                data[f'temp_rolling_max_{window}'] - data[f'temp_rolling_min_{window}']
        )
        # Медиана
        data[f'temp_rolling_median_{window}'] = data['temp'].rolling(window=window, min_periods=1).median()
        # Экспоненциальное взвешенное среднее
        data[f'temp_rolling_ewm_{window}'] = data['temp'].ewm(span=window, min_periods=1).mean()

    diffs = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720, 1080]  # час, день, неделя, месяц
    for diff in diffs:
        data[f'temp_diff_{diff}'] = data['temp'].diff(diff)
        # Процентное изменение
        data[f'temp_pct_change_{diff}'] = data['temp'].pct_change(diff)

    try:
        # Более гибкий метод STL
        stl = STL(data['time'], period=24, robust=True)
        result = stl.fit()

        data[f'time_trend'] = result.trend
        data[f'time_seasonal'] = result.seasonal
        data[f'time_residual'] = result.resid

        stl1 = STL(data['time'], period=24, robust=True)
        result = stl1.fit()
        residual1 = result.resid

        # Определение порогов для аномалий
        residual_mean = residual1.mean()
        residual_std = residual1.std()
        lower_threshold = residual_mean - 2 * residual_std
        upper_threshold = residual_mean + 2 * residual_std

        # Маркировка аномалий
        data[f'temp_is_anomaly'] = (
                (residual1 < lower_threshold) | (residual1 > upper_threshold)
        ).astype(int)

        # Добавление абсолютного значения отклонения (для ранжирования аномалий)
        data[f'temp_anomaly_score'] = np.abs((residual1 - residual_mean) / residual_std)

    except:
        # Если разложение не получается, заполняем нулями
        data[f'data_trend'] = 0
        data[f'data_seasonal'] = 0
        data[f'data_residual'] = 0
        data[f'temp_is_anomaly'] = 0
        data[f'temp_anomaly_score'] = 0

    periods = [24, 72, 168, 720]  # день, неделя, 30 дней
    harmonics = 3

    # Получаем временной индекс как числовую последовательность
    time_idx = np.arange(start=134736, stop=134736 + len(data))
    data['time_idx'] = time_idx

    # Создание признаков Фурье
    for period in periods:
        for harmonic in range(1, harmonics + 1):
            # Синус
            data[f'temp_fourier_sin_{period}_{harmonic}'] = np.sin(2 * np.pi * harmonic * data['temp'] / period)
            # Косинус
            data[f'temp_fourier_cos_{period}_{harmonic}'] = np.cos(2 * np.pi * harmonic * data['temp'] / period)

    # Создание взаимодействий признаков
    target_col = 'temp'
    if 'temperature_trend' in data.columns:
        data = create_feature_interaction(data, target_col, 'temperature_trend', operation='subtract')

    # Взаимодействие между сезонными признаками и температурой
    if 'month_sin' in data.columns:
        data = create_feature_interaction(data, target_col, 'month_sin', operation='multiply')

    if 'hour_sin' in data.columns:
        data = create_feature_interaction(data, target_col, 'hour_sin', operation='multiply')

    # Вычисление дополнительных статистик
    if f'{target_col}_rolling_mean_24' in data.columns and f'{target_col}_rolling_std_24' in data.columns:
        # Коэффициент вариации (отношение стандартного отклонения к среднему)
        data[f'{target_col}_cv_24'] = data[f'{target_col}_rolling_std_24'] / (
                data[f'{target_col}_rolling_mean_24'] + 1e-10)

    if f'{target_col}_rolling_max_24' in data.columns and f'{target_col}_rolling_min_24' in data.columns:
        # Нормализованный размах
        data[f'{target_col}_norm_range_24'] = (data[f'{target_col}_rolling_max_24'] - data[
            f'{target_col}_rolling_min_24']) / (data[f'{target_col}_rolling_mean_24'] + 1e-10)

    # Добавление флага аномально высокой или низкой температуры
    if f'{target_col}_is_anomaly' in data.columns:
        # Получаем статистики для сезонов
        if 'season' in data.columns:
            season_stats = data.groupby('season')[target_col].agg(['mean', 'std'])

            # Создаем новые признаки
            for season in data['season'].unique():
                season_mean = season_stats.loc[season, 'mean']
                season_std = season_stats.loc[season, 'std']

                mask = data['season'] == season
                data.loc[mask, f'{target_col}_season_zscore'] = (data.loc[mask, target_col] - season_mean) / (
                        season_std + 1e-10)
    data.dropna(inplace=True)

    last_index = data['time_idx'].max()

    new_df = pd.DataFrame(
        {'time': pd.date_range(start=pd.to_datetime(data['time'].iloc[-1]) + datetime.timedelta(hours=1),
                               periods=744, freq='H')})

    new_df['hour'] = new_df['time'].dt.hour
    new_df['day'] = new_df['time'].dt.day
    new_df['month'] = new_df['time'].dt.month
    new_df['dayofweek'] = new_df['time'].dt.dayofweek
    new_df['season'] = (new_df['month'] % 12 // 3).astype(int)

    new_df['hour_sin'] = np.sin(2 * np.pi * new_df['hour'] / 24)
    new_df['hour_cos'] = np.cos(2 * np.pi * new_df['hour'] / 24)

    new_df['day_sin'] = np.sin(2 * np.pi * new_df['day'] / 31)
    new_df['day_cos'] = np.cos(2 * np.pi * new_df['day'] / 31)

    new_df['month_sin'] = np.sin(2 * np.pi * new_df['month'] / 12)
    new_df['month_cos'] = np.cos(2 * np.pi * new_df['month'] / 12)

    new_df['dayofweek_sin'] = np.sin(2 * np.pi * new_df['dayofweek'] / 7)
    new_df['dayofweek_cos'] = np.cos(2 * np.pi * new_df['dayofweek'] / 7)

    new_df['season_sin'] = np.sin(2 * np.pi * new_df['season'] / 4)
    new_df['season_cos'] = np.cos(2 * np.pi * new_df['season'] / 4)

    new_df['time_idx'] = range(last_index + 1, last_index + 1 + len(new_df))

    new_df.fillna(0, inplace=True)

    df = pd.concat([data, new_df])

    df.fillna(0, inplace=True)

    df['station_id'] = 0

    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)

    return df


def get_prediction(data, dataset_path, model_path):
    if os.path.exists(dataset_path) and os.path.exists(model_path):

        with open(dataset_path, 'rb') as file:
            datas = pickle.load(file)

        model = TemporalFusionTransformer.load_from_checkpoint(model_path)

    else:
        raise FileNotFoundError

    dataset = TimeSeriesDataSet.from_parameters(parameters=datas, data=data, predict=True, stop_randomization=True)
    dataloader = dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    preds = model.predict(dataloader)

    point_predictions = preds.cpu().numpy().squeeze()

    predicted_values, _ = correct_adjustment(data, point_predictions)

    dates = data.iloc[-744:]['time'].values
    predicted_data = pd.DataFrame({'time': dates, 'temp': predicted_values})
    predicted_data = pd.concat([data[['time', 'temp']].iloc[-745:-744], predicted_data.iloc[:672]])
    predicted_data['time'] = pd.to_datetime(predicted_data['time'])
    return predicted_data
