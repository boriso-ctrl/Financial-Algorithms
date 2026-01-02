import warnings
import simfin as sf
import pandas as pd

# ==============================
# SimFin configuration
# ==============================

sf.set_api_key("985f143a-6709-4578-b2ee-4d5eaac01330")
sf.set_data_dir("C:/Users/boris/.simfin")

MARKET = "us"


# ==============================
# Data loading functions
# ==============================

def load_daily_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Load daily close prices for a list of tickers.

    Returns:
        DataFrame with:
        - index: Date (datetime)
        - columns: Tickers
        - values: Close prices
    """

    # Load full daily price dataset (cached after first run)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*date_parser.*",
            category=FutureWarning,
        )
        df = sf.load_shareprices(
            variant="daily",
            market=MARKET
        )

    # Filter to desired tickers
    df = df.loc[tickers]

    # Ensure clean structure
    df = (
        df[["Close"]]
        .reset_index()
        .drop_duplicates(subset=["Ticker", "Date"])
        .pivot(index="Date", columns="Ticker", values="Close")
        .sort_index()
    )

    return df


def load_daily_returns(tickers: list[str]) -> pd.DataFrame:
    """
    Load daily percentage returns for a list of tickers.
    """

    prices = load_daily_prices(tickers)
    returns = prices.pct_change().dropna()

    return returns
