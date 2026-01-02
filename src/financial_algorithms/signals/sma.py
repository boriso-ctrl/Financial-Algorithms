import numpy as np
import pandas as pd


def sma_signal(prices, fast=50, slow=200):
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()

    signal = np.sign(sma_fast - sma_slow)
    return signal.fillna(0)
