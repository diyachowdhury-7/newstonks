from transformers import pipeline
import pandas as pd

# Module-level variable to hold the model
# We load it once and reuse it — FinBERT is ~400MB
_sentiment_pipeline = None

def get_pipeline():
    """
    Loads FinBERT once and caches it at module level.
    Subsequent calls return the already-loaded model.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("Loading FinBERT model... (first time only)")
        _sentiment_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,      # headlines over 512 tokens get cut off
            max_length=512
        )
    return _sentiment_pipeline


def score_headline(headline: str) -> float:
    """
    Scores a single headline.

    Returns a float between -1.0 and +1.0:
      positive label → +confidence score
      negative label → -confidence score
      neutral  label → 0.0
    """
    pipe = get_pipeline()
    result = pipe(headline)[0]   # returns list of dicts, take first

    label = result["label"]      # "positive", "negative", or "neutral"
    score = result["score"]      # confidence, 0.0 to 1.0

    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0.0


def score_all_headlines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the headlines dataframe and adds a sentiment_score column.
    Then aggregates by date — one score per day.

    Args:
        df: DataFrame with columns: date, headline

    Returns:
        DataFrame with columns: date, sentiment_score
        One row per date, score is the mean of all headlines that day.
    """
    # Score every headline (this is the slow step — FinBERT runs on each one)
    df = df.copy()
    df["sentiment_score"] = df["headline"].apply(score_headline)

    # Group by date and take the mean sentiment for that day
    daily = (
        df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
    )

    daily = daily.sort_values("date").reset_index(drop=True)

    return daily