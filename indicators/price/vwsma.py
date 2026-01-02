"""VWAP-SMA mean reversion strategy."""

import pandas as pd
import ta


def vwsma_strategy(df: pd.DataFrame, sl: float) -> pd.DataFrame:
    """Annotate DataFrame with VWSMA signals in `VWSMA_signal`."""
    df = df.copy()
    df['VWAP'] = ta.volume.volume_weighted_moving_average(df['Close'], df['Volume'], n=20)
    df['Close-VWAP'] = df['Close'] - df['VWAP']
    df['zcore1'] = (
        (df['Close-VWAP'] - df['Close-VWAP'].rolling(window=40).mean())
        / (df['Close-VWAP'].rolling(window=40).std())
    )

    zcore = list(df['zcore1'])
    vwsma_signal = [0] * len(df)
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if df['Trend'][i] == 'Downtrend':
            if zcore[i] > 1 and zcore[i - 1] < 1:
                vwsma_signal[i] = -1
                count = i + 1
                if count < len(df):
                    cur_close = close[count - 1]
                    new_close = close[count]
                    pend_ret = (-1) * (new_close - cur_close) / cur_close
                    while sl < pend_ret and zcore[count] > 0 and count < len(df) - 1:
                        vwsma_signal[count] = -1
                        count += 1
                        cur_close = close[count - 1]
                        new_close = close[count]
                        x = (-1) * (new_close - cur_close) / cur_close
                        pend_ret += x
                    if sl < pend_ret and count == len(df) - 1 and zcore[count] > 0:
                        vwsma_signal[count] = -1
                i = count
            else:
                i += 1
        elif df['Trend'][i] == 'Uptrend':
            if zcore[i] < -1.5 and zcore[i - 1] > -1.5:
                vwsma_signal[i] = 1
                count = i + 1
                if count < len(df):
                    cur_close = close[count - 1]
                    new_close = close[count]
                    pend_ret = (new_close - cur_close) / cur_close
                    while zcore[count] < 1.5 and count < len(df) - 1:
                        vwsma_signal[count] = 1
                        count += 1
                        cur_close = close[count - 1]
                        new_close = close[count]
                        x = (new_close - cur_close) / cur_close
                        pend_ret += x
                    if count == len(df) - 1 and zcore[count] < 1.5:
                        vwsma_signal[count] = 1
                i = count
            else:
                i += 1
        else:
            if zcore[i] < -2 and zcore[i - 1] > -2:
                vwsma_signal[i] = 1
                count = i + 1
                if count < len(df):
                    cur_close = close[count - 1]
                    new_close = close[count]
                    pend_ret = (new_close - cur_close) / cur_close
                    while sl < pend_ret and zcore[count] < -1 and count < len(df) - 1:
                        vwsma_signal[count] = 1
                        count += 1
                        cur_close = close[count - 1]
                        new_close = close[count]
                        x = (new_close - cur_close) / cur_close
                        pend_ret += x
                    if sl < pend_ret and count == len(df) - 1 and zcore[count] < -1:
                        vwsma_signal[count] = 1
                i = count
            elif zcore[i] > 2 and zcore[i - 1] < 2:
                vwsma_signal[i] = -1
                count = i + 1
                if count < len(df):
                    cur_close = close[count - 1]
                    new_close = close[count]
                    pend_ret = (-1) * (new_close - cur_close) / cur_close
                    while sl < pend_ret and zcore[count] > 1 and count < len(df) - 1:
                        vwsma_signal[count] = -1
                        count += 1
                        cur_close = close[count - 1]
                        new_close = close[count]
                        x = (-1) * (new_close - cur_close) / cur_close
                        pend_ret += x
                    if count == len(df) - 1 and sl < pend_ret and zcore[count] > 1:
                        vwsma_signal[count] = -1
                i = count
            else:
                i += 1

    df['VWSMA_signal'] = vwsma_signal
    return df
