"""ADX-based trend strategy."""

import pandas as pd
import ta


def adx_strategy(df: pd.DataFrame, sl: float) -> pd.DataFrame:
    """Annotate DataFrame with ADX signals in `ADX_signal`."""
    df = df.copy()
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], n=14)

    adx_signal = [0] * len(df)
    adx = list(df['ADX'])
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if df['Trend'][i] == 'Uptrend':
            if adx[i] > 25:
                adx_signal[i] = 1
                count = i + 1
                if count < len(df):
                    cur_close = close[i]
                    new_close = close[count]
                    pend_ret = (new_close - cur_close) / cur_close
                    while sl < pend_ret and adx[count] > 20 and count < len(df) - 1:
                        adx_signal[count] = 1
                        count += 1
                        new_close = close[count]
                        pend_ret = (new_close - cur_close) / cur_close
                    if count == len(df) - 1 and adx[count] > 20 and sl < pend_ret:
                        adx_signal[count] = 1
                i = count
            else:
                i += 1
        elif df['Trend'][i] == 'Downtrend':
            if adx[i] > 25:
                adx_signal[i] = -1
                count = i + 1
                if count < len(df):
                    cur_close = close[i]
                    new_close = close[count]
                    pend_ret = (-1) * (new_close - cur_close) / cur_close
                    while sl < pend_ret and adx[count] > 25 and count < len(df) - 1:
                        adx_signal[count] = -1
                        count += 1
                        new_close = close[count]
                        pend_ret = (-1) * (new_close - cur_close) / cur_close
                    if count == len(df) - 1 and adx[count] > 25 and sl < pend_ret:
                        adx_signal[count] = -1
                i = count
            else:
                i += 1
        else:
            i += 1

    df['ADX_signal'] = adx_signal
    return df
