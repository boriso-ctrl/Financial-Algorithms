from data_loader import load_daily_prices
from signals.sma_signal import sma_signal

tickers = ["AAPL"]
prices = load_daily_prices(tickers)

signal = sma_signal(prices["AAPL"])
print(signal.tail())
