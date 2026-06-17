import os
import requests
import pandas as pd

os.environ["ALPHA_VANTAGE_KEY"] = "8W7OKNYYHER4IH6B"

def fetch_headlines(ticker: str, start: str, end: str) -> pd.DataFrame:
    api_key = os.getenv("ALPHA_VANTAGE_KEY")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": start.replace("-", "") + "T0000",
        "time_to": end.replace("-", "") + "T2359",
        "limit": 1000,
        "apikey": api_key
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    feed = data.get("feed", [])
    if not feed:
        raise ValueError(f"No news found for '{ticker}' in this date range.")

    rows = []
    for item in feed:
        title = item.get("title")
        pub = item.get("time_published")  # format: 20240101T120000
        if title and pub:
            rows.append({
                "date": pd.to_datetime(pub, format="%Y%m%dT%H%M%S").date(),
                "headline": title
            })

    df = pd.DataFrame(rows).dropna()
    df = df.sort_values("date").reset_index(drop=True)
    return df