"""Volume-Weighted Moving Average (VWAP-style)."""

import numpy as np
import pandas as pd


def volume_weighted_moving_average(close: pd.Series, volume: pd.Series, n: int = 20, fillna: bool = False) -> pd.Series:
    vwma = pd.Series([np.nan] * len(close), index=close.index)
    for i in range(n, len(close)):
        vwma[i] = np.average(list(close[i - n:i]), weights=list(volume[i - n:i]))
    if fillna:
        vwma = vwma.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(vwma, name='VWAP')
