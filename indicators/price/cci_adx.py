"""CCI + ADX combined strategy."""

import pandas as pd
import ta


def cci_adx_strategy(df: pd.DataFrame, sl: float) -> pd.DataFrame:
    """Annotate DataFrame with CDX signals in `CDX_signal`."""
    df = df.copy()
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], n=20, c=0.015)
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], n=14)

    cdx_signal = [0] * len(df)
    cci = list(df['CCI'])
    adx = list(df['ADX'])
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if adx[i] < 25:
            if df['Trend'][i] != 'Downtrend':
                if cci[i] < 100 and cci[i - 1] > 100:
                    cdx_signal[i] = 1
                    count = i + 1
                    if count < len(df):
                        cur_close = close[i]
                        new_close = close[count]
                        pend_ret = (new_close - cur_close) / cur_close
                        while sl < pend_ret and cci[count] > -100 and count < len(df) - 1:
                            cdx_signal[count] = 1
                            count += 1
                            new_close = close[count]
                            pend_ret = (new_close - cur_close) / cur_close
                        if count == len(df) - 1 and cci[count] > -100 and sl < pend_ret:
                            cdx_signal[count] = 1
                    i = count
                else:
                    i += 1
            else:
                if cci[i] > -100 and cci[i - 1] < -100:
                    cdx_signal[i] = -1
                    count = i + 1
                    if count < len(df):
                        cur_close = close[i]
                        new_close = close[count]
                        pend_ret = (-1) * (new_close - cur_close) / cur_close
                        while sl < pend_ret and cci[count] < 100 and count < len(df) - 1:
                            cdx_signal[count] = -1
                            count += 1
                            new_close = close[count]
                            pend_ret = (-1) * (new_close - cur_close) / cur_close
                        if count == len(df) - 1 and cci[count] < 100 and sl < pend_ret:
                            cdx_signal[count] = -1
                    i = count
                else:
                    i += 1
        else:
            i += 1

    df['CDX_signal'] = cdx_signal
    return df
