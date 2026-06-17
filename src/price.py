import yfinance as yf
import pandas as pd


def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads daily OHLCV data for a ticker and computes daily % return.

    Args:
        ticker: e.g. "AAPL" or "RELIANCE.NS"
        start:  "YYYY-MM-DD"
        end:    "YYYY-MM-DD"

    Returns:
        DataFrame with columns: date (datetime.date), close, pct_change
    """
    raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'.")

    df = pd.DataFrame()
    df["close"] = raw["Close"].squeeze()   # .squeeze() handles multi-index edge case

    # pct_change() computes (today - yesterday) / yesterday * 100
    # First row will always be NaN since there's no previous day — that's fine
    df["pct_change"] = df["close"].pct_change() * 100

    # Convert index (which is datetime) to plain date for merging later
    df["date"] = df.index.date
    df = df[["date", "close", "pct_change"]].dropna()
    df = df.reset_index(drop=True)

    return df