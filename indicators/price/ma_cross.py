"""Moving-average crossover strategy (MA20/MA50 by default)."""

import pandas as pd

def ma_cross_strategy(df: pd.DataFrame, sl: float, n1: int = 20, n2: int = 50) -> pd.DataFrame:
    """Annotate DataFrame with MA20/MA50 signals in `MA_signal` column."""
    df = df.copy()
    df['MA20'] = df['Close'].rolling(window=n1).mean()
    df['MA50'] = df['Close'].rolling(window=n2).mean()

    ma_signal = [0] * len(df)
    ma20 = list(df['MA20'])
    ma50 = list(df['MA50'])
    close = list(df['Close'])

    i = max(n1, n2) + 1
    while i < len(df):
        if df['Trend'][i] == 'Uptrend':
            if ma20[i] >= ma50[i]:
                ma_signal[i] = 1
                count = i + 1
                if count < len(df):
                    cur_close = close[count - 1]
                    new_close = close[count]
                    pend_ret = (new_close - cur_close) / cur_close
                    while sl < pend_ret and ma20[count] > ma50[count] and count < len(df) - 1:
                        ma_signal[count] = 1
                        count += 1
                        cur_close = close[count - 1]
                        new_close = close[count]
                        x = (new_close - cur_close) / cur_close
                        pend_ret += x
                    if count == len(df) - 1 and ma20[count] > ma50[count] and sl < pend_ret:
                        ma_signal[count] = 1
                i = count
            else:
                i += 1
        elif df['Trend'][i] == 'Downtrend':
            if ma20[i] <= ma50[i]:
                ma_signal[i] = -1
                count = i + 1
                if count < len(df):
                    cur_close = close[count - 1]
                    new_close = close[count]
                    pend_ret = (-1) * (new_close - cur_close) / cur_close
                    while sl < pend_ret and ma20[count] < ma50[count] and count < len(df) - 1:
                        ma_signal[count] = -1
                        count += 1
                        cur_close = close[count - 1]
                        new_close = close[count]
                        x = (-1) * (new_close - cur_close) / cur_close
                        pend_ret += x
                    if count == len(df) - 1 and ma20[count] < ma50[count] and sl < pend_ret:
                        ma_signal[count] = -1
                i = count
            else:
                i += 1
        else:
            i += 1

    df['MA_signal'] = ma_signal
    return df
