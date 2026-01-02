"""On-Balance Volume (OBV)."""

import numpy as np
import pandas as pd


def on_balance_volume(close: pd.Series, volume: pd.Series, fillna: bool = False) -> pd.Series:
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = np.nan
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = -volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    obv = df['OBV'].cumsum()
    if fillna:
        obv = obv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(obv, name='obv')
