import datetime
import time
from logging import config

import numpy as np
import pandas as pd
from autots import AutoTS
from twisted.application.internet import ClientService

CONFIG_FILE = "settings.ini"

last_result = None
last_execution_time = None
client_service = ClientService(config.tinkoff_token, config.tinkoff_app_name)
tickers = ['SBER', 'GAZP', 'ROSN', 'LKOH']
num_cycles = 1
flat_level = 30


def get_pair_dfs(tickers, num_cycles):
    tickers_dfs = {}
    for ticker in tickers:
        df = pd.DataFrame(columns=['Opentime', f'{ticker}_Open', f'{ticker}_High', f'{ticker}_Low', f'{ticker}_Close'])
        for i in range(num_cycles):
            tickers_data = client_service.klines(symbol=ticker, interval=client_service.KLINE_INTERVAL_1HOUR, limit=500)
            if not tickers_data:
                break
            temp_df = pd.DataFrame(tickers_data,
                                   columns=['Opentime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Closetime', 'Quotee',
                                            'Number', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
            if temp_df.empty:
                break
            temp_df = temp_df[['Opentime', 'Open', 'High', 'Low', 'Close']]
            temp_df.columns = ['Opentime', f'{ticker}_Open', f'{ticker}_High', f'{ticker}_Low', f'{ticker}_Close']
            df = pd.concat([df, temp_df], ignore_index=True)
            end_time = int(temp_df.Opentime[0] - 1)
            # print(f'Цикл {i} для {ticker}: {len(futures_data)} записей добавлено.')
            time.sleep(2)
        tickers_dfs[ticker] = df
    # print(tickers_dfs)
    return tickers_dfs


def combine_pair_dfs(pair_dfs):
    combined_df = pd.DataFrame()
    for pair, df in pair_dfs.items():
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df.iloc[:, 1:]], axis=1)
    combined_df['Opentime'] = pd.to_datetime(combined_df['Opentime'], unit='ms')
    combined_df = combined_df.set_index('Opentime')
    combined_df = combined_df.iloc[:, 1:].astype(float)
    df_diff = combined_df.diff()
    df_diff['target'] = df_diff['SBER_High'] + df_diff['SBER_Low']
    df_diff = df_diff.drop(df_diff.index[0])
    df_diff = df_diff.reset_index()
    # print(df_diff)
    return df_diff.iloc[1:]


def train_models(df_diff):
    # Создание модели AutoTS для столбца 'SBER_High'
    model_high = AutoTS(
        forecast_length=3,
        frequency='infer',
        ensemble='bestn',
        model_list="fast",
        transformer_list="fast",
        # drop_most_recent=1,
        max_generations=5,
        num_validations=3,
        validation_method="backwards"
    )
    # Обучение модели AutoTS для столбца 'target'
    model_high = model_high.fit(df_diff, date_col="Opentime", value_col='target')
    # Предсказание
    prediction_high = model_high.predict()
    # Прогнозные данные
    forecast_high = prediction_high.forecast
    print(forecast_high)
    # Дозаписывание прогнозных данных в текстовый файл
    with open("model_high.txt", 'a') as file:
        file.write(f"{model_high}\n")
    return forecast_high


def get_trend_direction(forecast_High):
    trend_direction_High = np.where(
        forecast_High['target'] > flat_level, 2, np.where(forecast_High['target'] < -flat_level, 1, 0))
    print(trend_direction_High)
    return trend_direction_High


def get_value(trend_direction):
    if np.all(trend_direction == 1):
        # print('Продаем')
        return 1
    elif np.all(trend_direction == 2):
        # print('Покупаем')
        return 2
    else:
        # print('Ждем')
        return 0


def run_once_per_hour():
    global last_execution_time, result
    current_time = datetime.datetime.now().replace(second=0, microsecond=0)
    if last_execution_time is None or current_time > last_execution_time + datetime.timedelta(hours=1):
        last_execution_time = current_time
        pair_dfs = get_pair_dfs(tickers, num_cycles)
        df = combine_pair_dfs(pair_dfs)
        forecast_High = train_models(df)
        trend_direction = get_trend_direction(forecast_High)
        result = get_value(trend_direction)
    # print('Сигнал от AutoTS:', result)
    return result
