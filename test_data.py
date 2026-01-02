from data_loader import load_daily_prices

tickers = ["AAPL", "MSFT", "AMZN"]

df = load_daily_prices(tickers)
print(df.tail())
