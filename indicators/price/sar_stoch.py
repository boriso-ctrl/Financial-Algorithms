"""Parabolic SAR + Stochastic oscillator strategy."""

import pandas as pd
import ta


def sar_stoch_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate DataFrame with combined SAR/Stochastic signals in `SS_signal`."""
    df = df.copy()
    df['SAR'] = ta.trend.sar(df, af=0.02, amax=0.2)
    df['Stoch%K'] = ta.momentum.stoch(pd.Series(df['High']), df['Low'], df['Close'], n=14)
    df['Stoch%D'] = ta.momentum.stoch_signal(pd.Series(df['High']), df['Low'], df['Close'], n=14)
    df['KminusD'] = df['Stoch%K'] - df['Stoch%D']

    sar = list(df['SAR'])
    stoch = list(df['Stoch%K'])
    kminusd = list(df['KminusD'])
    ss_signal = [0] * len(df)
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if df['Trend'][i] == 'Uptrend':
            if sar[i] < close[i]:
                if (stoch[i] >= 20 and stoch[i - 1] <= 20) or (kminusd[i] > 0 and kminusd[i - 1] < 0):
                    ss_signal[i] = 1
                    count = i + 1
                    if count < len(df):
                        cur_close = close[i]
                        new_close = close[count]
                        slsar = (sar[i] - cur_close) / cur_close
                        pend_ret = (new_close - cur_close) / cur_close
                        while slsar < pend_ret and sar[count] < close[count] and count < len(df) - 1:
                            ss_signal[count] = 1
                            slsar = (sar[count] - cur_close) / cur_close
                            count += 1
                            new_close = close[count]
                            pend_ret = (new_close - cur_close) / cur_close
                        if count == len(df) - 1 and sar[count] < close[count] and (sar[count] - cur_close) / cur_close < pend_ret:
                            ss_signal[count] = 1
                    i = count
                else:
                    i += 1
            else:
                i += 1

        if df['Trend'][i] == 'Downtrend':
            if sar[i] > close[i]:
                if (stoch[i] <= 80 and stoch[i - 1] >= 80) or (kminusd[i] < 0 and kminusd[i - 1] > 0):
                    ss_signal[i] = -1
                    count = i + 1
                    if count < len(df):
                        cur_close = close[i]
                        new_close = close[count]
                        slsar = (-1) * (sar[i] - cur_close) / cur_close
                        pend_ret = (-1) * (new_close - cur_close) / cur_close
                        while slsar < pend_ret and sar[count] > close[count] and count < len(df) - 1:
                            ss_signal[count] = -1
                            slsar = (-1) * (sar[count] - cur_close) / cur_close
                            count += 1
                            new_close = close[count]
                            pend_ret = (-1) * (new_close - cur_close) / cur_close
                        if count == len(df) - 1 and sar[count] > close[count] and (-1) * ((sar[count] - cur_close)) / cur_close < pend_ret:
                            ss_signal[count] = -1
                    i = count
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    df['SS_signal'] = ss_signal
    return df
