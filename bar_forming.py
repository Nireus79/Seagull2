import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import winsound

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# https://data.binance.vision/
def csv_merger(path, type_of_data):
    if type_of_data == 'time':
        names = glob.glob(path + "*.csv")  # get names of all CSV files under path
        # If your CSV files use commas to split fields, then the sep argument can be omitted or set to ","
        # columns = ['id', 'price', 'qty', 'base_qty', 'time', 'is_buyer_maker', '7'] # TICK columns
        columns = ['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open time',
                   '0', '1', '2', '3', '4']
        #     1499040000000,      // Open time # Klines columns
        #     "0.01634790",       // Open
        #     "0.80000000",       // High
        #     "0.01575800",       // Low
        #     "0.01577100",       // Close
        #     "148976.11427815",  // Volume
        #     1499644799999,      // Close time
        #     "2434.19055334",    // Quote asset volume
        #     308,                // Number of trades
        #     "1756.87402397",    // Taker buy base asset volume
        #     "28.46694368",      // Taker buy quote asset volume
        #     "17928899.62484339" // Ignore.
        data = pd.concat([pd.read_csv(filename, names=columns, sep=",") for filename in names])
        data.drop(columns=['Open time', '0', '1', '2', '3', '4'], axis=1, inplace=True)
        data['time'] = data['time'].apply(lambda x: x // 1000 if x >= 10 ** 15 else x)
        # save the DataFrame to a file
        data.to_csv("ETHEUR_5m.csv", index=False)
        return data
    elif type_of_data == 'tick':
        names = glob.glob(path + "*.csv")  # get names of all CSV files under path
        # If your CSV files use commas to split fields, then the sep argument can be omitted or set to ","
        # columns = ['id', 'price', 'qty', 'base_qty', 'time', 'is_buyer_maker', '7'] # TICK columns
        columns = ['Close', 'Quantity', 'Value', 'time', '1', '2']
        data = pd.concat([pd.read_csv(filename, names=columns, sep=",") for filename in names])
        data['time'] = data['time'].apply(lambda x: x // 1000 if x >= 10 ** 15 else x)
        # save the DataFrame to a file
        data.to_csv("ETHEUR_5m.csv", index=False)
        return data


print(csv_merger('D:/Seagull_data/historical_data/time/ETHEUR/5m/', 'time'))


def mad_outlier(y, thresh=3.):
    """
    compute outliers based on mad
    # args
        y: assumed to be arrayed with shape (N,1)
        thresh: float()
    # returns
        array index of outliers
    """
    median = np.median(y)
    diff = np.sum((y - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def returns(s):
    arr = np.diff(np.log(s))
    return pd.Series(arr, index=s.index[1:])


def tick_bars(df, price_column, m):
    """
    compute tick bars
    # args
        df: pd.DataFrame()
        column: name for price data
        m: int(), threshold value for ticks
    # returns
        idx: list of indices
    """
    t = df[price_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += 1
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx


def tick_bar_df(df, price_column, m):
    idx = tick_bars(df, price_column, m)
    return df.iloc[idx]


# ========================================================
def volume_bars(df, volume_column, m):
    """
    compute volume bars
    # args
        df: pd.DataFrame()
        column: name for volume data
        m: int(), threshold value for volume returns
        idx: list of indices
    """
    t = df[volume_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx


def volume_bar_df(df, volume_column, m):
    idx = volume_bars(df, volume_column, m)
    return df.iloc[idx]


# ========================================================
def dollar_bars(df, value_column, m):
    """
    compute dollar bars
    # args
        df: pd.DataFrame()
        column: name for dollar volume data
        m: int(), threshold value for dollars returns
        idx: list of indices
    """
    t = df[value_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx


def dollar_bar_df(df, value_column, m):
    idx = dollar_bars(df, value_column, m)
    return df.iloc[idx]


def form_dollar_bars(csv, vol):
    data = pd.read_csv(csv)
    # data.time = pd.to_datetime(data.time, unit='ms')
    # data.set_index('time', inplace=True)
    # mad = mad_outlier(data.price.values.reshape(-1, 1))
    # data = data.loc[~mad]
    # print(data)
    dbars = dollar_bar_df(data, 'Value', vol).dropna()
    # dbars.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    dbars.to_csv('ETHEUR_1mdb.csv')
    return dbars


# print(form_dollar_bars('ETHEUR_full_tick.csv', 1000000))


def form_time_bars(csv, frequency):
    """
    Takes tick to tick data frame and structures data by given time freq.
    param data:
    :param csv:
    :param frequency: string ex: '5min'
    see doc https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    :return: time bars of given frequency
    Time bars oversample information during low-activity periods and undersample
    information during high-activity periods.
    Time-sampled series often exhibit poor
    statistical properties, like serial correlation, heteroscedasticity, and non-normality of
    returns.
    WARNING (works with datetime index only datetime_indexing function must be used as arg)
    WARNING Lopez de Prado suggests 1min bars as substitute for constructing market microstructural futures
    """
    data = pd.read_csv(csv)
    data.time = pd.to_datetime(data.time, unit='ms')
    data.set_index('time', inplace=True)
    data.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    data = data.loc[~data.index.duplicated(keep='first')]
    time_bars = data.groupby(pd.Grouper(freq=frequency)).agg({'price': 'ohlc', 'qty': 'sum'})
    time_bars_price = time_bars.loc[:, 'price']
    time_bars_price.ffill(inplace=True)
    time_bars_price.to_csv('ETHEUR_5m.csv')
    return time_bars_price
