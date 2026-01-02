import simfin as sf
import pandas as pd

# -----------------------------
# Settings
# -----------------------------
sf.set_api_key('985f143a-6709-4578-b2ee-4d5eaac01330')  # Your SimFin API key
sf.set_data_dir('C:/Users/boris/.simfin')               # Local folder to store data

# -----------------------------
# Load US daily share prices
# -----------------------------
print("Loading US daily share prices (this may take a few minutes)...")
df_prices = sf.load_shareprices(variant='daily', market='us')
print("Data loaded.")

# -----------------------------
# Select subset of tickers
# -----------------------------
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
df_prices_daily = df_prices.loc[tickers].copy()

# -----------------------------
# Fix duplicates and reshape
# -----------------------------
# Take 'Close' prices, group by Ticker & Date, take last value, unstack Ticker
df_close = df_prices_daily['Close'].groupby(['Ticker', 'Date']).last().unstack(level=0)

# -----------------------------
# Compute daily returns
# -----------------------------
df_returns = df_close.pct_change().dropna()

# -----------------------------
# Compute summary statistics
# -----------------------------
volatility = df_returns.std()  # Daily std (volatility)
correlation = df_returns.corr()  # Correlation matrix

# -----------------------------
# Print results
# -----------------------------
print("\n--- Summary Statistics ---")
print("Volatility (daily std of returns):")
print(volatility)
print("\nCorrelation between stocks:")
print(correlation)
