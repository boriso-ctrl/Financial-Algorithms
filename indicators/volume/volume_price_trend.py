"""Volume-Price Trend (VPT)."""

import numpy as np
import pandas as pd


def volume_price_trend(close: pd.Series, volume: pd.Series, fillna: bool = False) -> pd.Series:
    vpt = volume * ((close - close.shift(1, fill_value=close.mean())) / close.shift(1, fill_value=close.mean()))
    vpt = vpt.shift(1, fill_value=vpt.mean()) + vpt
    if fillna:
        vpt = vpt.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(vpt, name='vpt')
