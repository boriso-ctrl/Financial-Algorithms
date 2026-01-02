"""Ease of Movement (EoM/EMV)."""

import numpy as np
import pandas as pd


def ease_of_movement(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 20, fillna: bool = False) -> pd.Series:
    emv = (high.diff(1) + low.diff(1)) * (high - low) / (2 * volume)
    emv = emv.rolling(n, min_periods=0).mean()
    if fillna:
        emv = emv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(emv, name='eom_' + str(n))
