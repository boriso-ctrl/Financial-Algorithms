"""Volume Oscillator."""

import numpy as np
import pandas as pd


def volume_oscillator(volume: pd.Series, s: int = 9, l: int = 26, fillna: bool = False) -> pd.Series:
    emas = volume.rolling(s, min_periods=0).mean()
    emal = volume.rolling(l, min_periods=0).mean()
    vo = (emas - emal) * 100 / emal
    if fillna:
        vo = vo.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(vo, name='VO')
