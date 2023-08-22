import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tinkoff.invest import Client

CONFIG_FILE = "../../settings.ini"
TICKER = 'SBER'  # Цифровой актив
TOKEN = os.environ["INVEST_TOKEN"]
client = Client(TOKEN)


# Получение Данных
def data_fr():
    futures_data = client.klines(symbol=TICKER, interval=client.KLINE_INTERVAL_5MINUTE, limit=100)
    temp_df = pd.DataFrame(futures_data, columns=['Opentime', 'Open', 'High', 'Low', 'Close', 'Volume',
                                                  'Closetime', 'Quotee',
                                                  'Number', 'TakerBuyBaseAssetVolume',
                                                  'TakerBuyQuoteAssetVolume', 'Ignore'])
    df = temp_df.loc[:, ['Opentime', 'Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    return df


# Подготовка данных для индикатора
def PrepareDF(DF):
    ohlc = DF.iloc[:, [0, 1, 2, 3, 4, 5]]
    ohlc.columns = ["date", "open", "high", "low", "close", "volume"]
    ohlc = ohlc.set_index('date')
    df = indBollingerBands(ohlc).reset_index()
    df['slope_up'] = indSlope(df['high'], 9)
    df['slope_low'] = indSlope(df['low'], 9)
    df['dif_up'] = df['high'].diff()
    df['dif_low'] = df['low'].diff()
    df['position_up'] = (df['high'] - df['upper'])
    df['position_low'] = (df['low'] - df['lower'])
    df = df.set_index('date')
    df = df.reset_index()
    return df


# Находим наклон ценовой линии для индикатора
def indSlope(series, n):
    array_sl = [j * 0 for j in range(n - 1)]

    for j in range(n, len(series) + 1):
        y = series[j - n:j]
        x = np.array(range(n))
        x_sc = (x - x.min()) / (x.max() - x.min())
        y_sc = (y - y.min()) / (y.max() - y.min())
        x_sc = sm.add_constant(x_sc)
        model = sm.OLS(y_sc, x_sc)
        results = model.fit()
        array_sl.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(array_sl))))
    return np.array(slope_angle)


# Расчет True Range и Average True Range индикатора
def indBollingerBands(source_DF, window=20, std_dev=3):
    df = pd.DataFrame(source_DF)
    # Рассчитываем скользящую среднюю
    df['sma'] = df['close'].rolling(window).mean()
    # Рассчитываем стандартное отклонение
    df['std'] = df['close'].rolling(window).std()
    # Рассчитываем верхнюю полосу Боллинджера
    df['upper'] = df['sma'] + (std_dev * df['std'])
    # Рассчитываем нижнюю полосу Боллинджера
    df['lower'] = df['sma'] - (std_dev * df['std'])
    # Ширина канала Болинджера
    df['chanal'] = df['upper'] - df['lower']
    return df


# локальный минимум/локальный максимум
def isLCC(DF, i):
    df = DF.copy()
    LCC = 0

    if df['low'][i] < df['low'][i + 1] and df['low'][i] < df['low'][i - 1] < df['low'][i + 1]:
        # найдено Дно
        LCC = i
    return LCC


def isHCC(DF, i):
    df = DF.copy()
    HCC = 0
    if df['high'][i] > df['high'][i + 1] and df['high'][i] > df['high'][i - 1] > df['high'][i + 1]:
        # найдена вершина
        HCC = i
    return HCC


# Функция для формирования сигнала для бота от индикатора
def check_if_signal():
    ohlc = data_fr()
    prepared_df = PrepareDF(ohlc)
    signalAtr = None

    i = len(prepared_df) - 1  # 99 - текущая незакрытая свеча, 98 - последняя закрытая свеча, нижняя или верхняя

    if isLCC(prepared_df, i - 1) > 0:
        print("Найдена дно")
        # найдено дно
        if prepared_df['position_low'][i - 1] < 10:
            # Пересечение нижней границы канала
            if prepared_df['chanal'][i - 1] > 100:
                # Ширина канала больше 100
                signalAtr = 2  # Если нет направления AutoTs то  Long

    if isHCC(prepared_df, i - 1) > 0:
        # найденная вершина
        if prepared_df['position_up'][i - 1] > 10:
            print("Найдена вершина")
            # Пересечение верхней границы канала
            if prepared_df['chanal'][i - 1] > 100:
                # Ширина канала больше 100
                signalAtr = 1  # Если нет направления AutoTs то Short
    # print('Сигнал от бота:', signalAtr)
    return signalAtr
