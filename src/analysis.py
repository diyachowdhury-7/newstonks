import pandas as pd
from scipy import stats


def compute_lag_correlations(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    For each lag offset from 0 to max_lag days, computes the Pearson
    correlation between sentiment (at that lag) and next-day price change.

    Lag 0: does today's sentiment correlate with today's return?
    Lag 1: does yesterday's sentiment correlate with today's return?
    ...and so on.

    Args:
        df:       Merged dataframe with sentiment_score and pct_change columns
        max_lag:  How many days back to test (default 5)

    Returns:
        DataFrame with columns:
          lag, correlation, p_value, significant (bool), n_samples
    """
    results = []

    for lag in range(0, max_lag + 1):
        # .shift(lag) moves the sentiment column DOWN by `lag` rows
        # So row i of shifted sentiment aligns with row i+lag of price
        # This means: sentiment from `lag` days ago vs today's price
        shifted_sentiment = df["sentiment_score"].shift(lag)

        # Build a temporary dataframe and drop rows where either is NaN
        # (shifting creates NaN at the top)
        temp = pd.DataFrame({
            "sentiment": shifted_sentiment,
            "pct_change": df["pct_change"]
        }).dropna()

        n = len(temp)

        # Need at least 3 data points to compute a meaningful correlation
        if n < 3:
            continue

        r, p = stats.pearsonr(temp["sentiment"], temp["pct_change"])

        results.append({
            "lag": lag,
            "lag_label": f"Lag {lag}d",
            "correlation": round(r, 3),
            "p_value": round(p, 3),
            "significant": p < 0.05,
            "n_samples": n
        })

    return pd.DataFrame(results)