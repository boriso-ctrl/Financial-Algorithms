import simfin as sf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Set API key and data dir
# ---------------------------
sf.set_api_key('985f143a-6709-4578-b2ee-4d5eaac01330')
sf.set_data_dir('C:/Users/boris/.simfin')

# ---------------------------
# 2. Load US daily share prices
# ---------------------------
print("Loading US daily share prices (this may take a few minutes)...")
df_prices = sf.load_shareprices(variant='daily', market='us')
print("Done!\n")

# ---------------------------
# 3. Select tickers
# ---------------------------
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
df_prices_daily = df_prices.loc[tickers].copy()

# ---------------------------
# 4. Prepare close prices (handle duplicates)
# ---------------------------
# Group by ticker & date, take last value to remove duplicates
df_close = df_prices_daily['Close'].groupby(['Ticker', 'Date']).last().unstack(level=0)

# ---------------------------
# 5. Compute daily returns
# ---------------------------
df_returns = df_close.pct_change().dropna()

# ---------------------------
# 6. Summary statistics
# ---------------------------
print("--- Daily Volatility per Ticker ---")
print(df_returns.std())

print("\n--- Correlation Between Stocks ---")
print(df_returns.corr())

# ---------------------------
# 7. Plot closing prices
# ---------------------------
plt.figure(figsize=(12, 6))
df_close.plot(title='Closing Prices of Selected US Stocks')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(title='Ticker')
plt.tight_layout()
plt.show()

# ---------------------------
# 8. Plot correlation heatmap
# ---------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df_returns.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Stocks')
plt.tight_layout()
plt.show()
