import pandas as pd

def sma_signal(prices, fast=50, slow=200):
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()

    signal = (sma_fast > sma_slow).astype(int)
    return signal
