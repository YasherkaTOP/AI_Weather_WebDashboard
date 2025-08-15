import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import datetime, timedelta, timezone
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm
from pytorch_forecasting import NaNLabelEncoder, TimeSeriesDataSet, GroupNormalizer
from config import BATCH_SIZE, OPEN_METEO_ARCHIVE_URL
from lightning.pytorch.tuner import Tuner
import pickle
import lightning.pytorch as pl


def get_dataloaders_and_model(objects):
    start_date = (datetime.now(timezone.utc) - timedelta(days=1096)).replace(day=1).strftime('%Y-%m-%d')
    end_date = (datetime.now(timezone.utc).replace(day=1, hour=0) - timedelta(hours=1)).strftime('%Y-%m-%d')
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    all_data = []

    for i in tqdm(range(len(objects))):
        obj = objects[i]
        lat = obj['lat']
        lng = obj['lng']

        params = {
            "latitude": float(lat),
            "longitude": float(lng),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m"],
            "timezone": "GMT",
        }

        responses = openmeteo.weather_api(OPEN_METEO_ARCHIVE_URL, params=params)
        hourly = responses[0].Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

        hourly_data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temp": hourly_temperature_2m,
        }
        data = pd.DataFrame(data=hourly_data)

        data['latitude'] = lat
        data['longitude'] = lng
        data['station_id'] = i
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

        data.drop(['hour', 'season'], axis=1, inplace=True)
        data.dropna(inplace=True)
        all_data.append((data.copy()))
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['time'] = pd.to_datetime(combined_df['time'])
    train = combined_df[combined_df['station_id'] != 0]
    test = combined_df[combined_df['station_id'] == 0]
    label_encoder = NaNLabelEncoder(add_nan=True).fit(train['station_id'])
    train_dataset = TimeSeriesDataSet(
        train,
        time_idx="time_idx",
        target='temp',
        group_ids=["station_id"],
        min_encoder_length=1440,
        min_prediction_length=672,
        max_encoder_length=1440,
        max_prediction_length=672,
        static_reals=['latitude', 'longitude'],
        time_varying_unknown_reals=['temp', 'temp_lag_1', 'temp_lag_24', 'temp_rolling_mean_168',
                                    'temp_rolling_std_168', 'temp_rolling_min_168', 'temp_rolling_max_168',
                                    'temp_rolling_mean_672', 'temp_rolling_std_672', 'temp_rolling_min_672',
                                    'temp_rolling_max_672', 'temp_diff_1', 'temp_pct_change_1', 'temp_diff_24',
                                    'temp_pct_change_24', 'temp_trend', 'temp_seasonal', 'temp_anomaly_score'],
        time_varying_known_reals=['hour_sin', 'hour_cos', 'month_sin', 'month_cos'],
        add_target_scales=True,
        add_relative_time_idx=True,
        target_normalizer=GroupNormalizer(
            groups=['station_id']
        ),
        categorical_encoders={"series": label_encoder}

    )
    test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, test, stop_randomization=True,
                                                  categorical_encoders={"series": label_encoder})

    with open('result/dataset_params.pickle', 'wb') as f:
        pickle.dump(test_dataset.get_parameters(), f)

    train_dataloader = train_dataset.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0
    )
    test_dataloader = test_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 10, num_workers=0
    )

    with open('result/label_encoder.pickle', 'wb') as f:
        pickle.dump(label_encoder, f)

    return train_dataloader, test_dataloader, train_dataset


def get_lr(model, train_dataloader, test_dataloader):
    pl.seed_everything(42)
    trainer = pl.Trainer(
        accelerator="auto",
        gradient_clip_val=0.1,
        limit_train_batches=100,
        limit_val_batches=20,
        max_epochs=1
    )

    res = Tuner(trainer).lr_find(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
        max_lr=1.0,
        min_lr=1e-8
    )
    return res.suggestion()
