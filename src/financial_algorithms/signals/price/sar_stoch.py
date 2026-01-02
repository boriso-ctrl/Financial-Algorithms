"""Parabolic SAR + Stochastic oscillator strategy with local PSAR fallback."""

import pandas as pd


def _parabolic_sar(high: pd.Series, low: pd.Series, af: float = 0.02, amax: float = 0.2) -> pd.Series:
    """Compute Parabolic SAR without external deps.

    Simplified standard implementation; returns a Series aligned to high/low.
    """
    if len(high) == 0:
        return pd.Series(dtype=float)

    af_step = af
    trend_up = True
    ep = high.iloc[0]
    sar = low.iloc[0]
    out = [sar]

    for i in range(1, len(high)):
        prev_sar = sar
        if trend_up:
            sar = prev_sar + af * (ep - prev_sar)
            sar = min(sar, low.iloc[i - 1], low.iloc[i - 2] if i > 1 else low.iloc[i - 1])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_step, amax)
            if low.iloc[i] < sar:
                trend_up = False
                sar = ep
                ep = low.iloc[i]
                af = af_step
        else:
            sar = prev_sar + af * (ep - prev_sar)
            sar = max(sar, high.iloc[i - 1], high.iloc[i - 2] if i > 1 else high.iloc[i - 1])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_step, amax)
            if high.iloc[i] > sar:
                trend_up = True
                sar = ep
                ep = high.iloc[i]
                af = af_step
        out.append(sar)

    return pd.Series(out, index=high.index, name='SAR')


def sar_stoch_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate DataFrame with combined SAR/Stochastic signals in `SS_signal`."""
    df = df.copy()
    df['SAR'] = _parabolic_sar(df['High'], df['Low'], af=0.02, amax=0.2)
    # Simple Stoch %K/%D
    low_roll = df['Low'].rolling(window=14)
    high_roll = df['High'].rolling(window=14)
    lowest_low = low_roll.min()
    highest_high = high_roll.max()
    denom = (highest_high - lowest_low).replace(0, pd.NA)
    df['Stoch%K'] = (df['Close'] - lowest_low) / denom * 100
    df['Stoch%D'] = df['Stoch%K'].rolling(window=3).mean()
    df['KminusD'] = df['Stoch%K'] - df['Stoch%D']

    sar = list(df['SAR'])
    stoch = list(df['Stoch%K'])
    kminusd = list(df['KminusD'])
    ss_signal = [0] * len(df)
    close = list(df['Close'])

    i = 51
    while i < len(df):
        if df['Trend'].iloc[i] == 'Uptrend':
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

        if df['Trend'].iloc[i] == 'Downtrend':
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
