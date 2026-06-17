import pandas as pd
import numpy as np


def roll_weekend_sentiment(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Markets are closed Saturday and Sunday, but news keeps happening.
    This function accumulates Saturday + Sunday sentiment and adds it
    to the following Monday's score.

    Args:
        sentiment_df: DataFrame with columns: date, sentiment_score

    Returns:
        Same structure but weekend rows removed and their scores
        rolled into the next Monday.
    """
    df = sentiment_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    rolled_rows = []
    weekend_accumulator = 0.0
    weekend_count = 0      # track how many days we're averaging over

    for _, row in df.iterrows():
        weekday = row["date"].weekday()   # 0=Mon, 1=Tue ... 5=Sat, 6=Sun

        if weekday >= 5:
            # Saturday or Sunday — accumulate, don't emit a row
            weekend_accumulator += row["sentiment_score"]
            weekend_count += 1
        else:
            # Weekday — add any accumulated weekend sentiment
            if weekend_count > 0:
                # Average the weekend scores in, weighted equally with today
                total_score = (row["sentiment_score"] + weekend_accumulator) / (1 + weekend_count)
                weekend_accumulator = 0.0
                weekend_count = 0
            else:
                total_score = row["sentiment_score"]

            rolled_rows.append({
                "date": row["date"].date(),
                "sentiment_score": round(total_score, 4)
            })

    return pd.DataFrame(rolled_rows)


def merge_sentiment_and_price(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins sentiment scores with price data on date.
    Only keeps days where both exist (inner join).

    Returns:
        DataFrame with columns: date, sentiment_score, close, pct_change
    """
    sentiment_rolled = roll_weekend_sentiment(sentiment_df)

    # Ensure both date columns are the same type before merging
    sentiment_rolled["date"] = pd.to_datetime(sentiment_rolled["date"])
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])

    merged = pd.merge(sentiment_rolled, price_df, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged