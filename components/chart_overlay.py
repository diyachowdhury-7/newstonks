import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_sentiment_vs_price(df: pd.DataFrame) -> plt.Figure:
    """
    Dual-axis chart:
      Left axis (bars):  daily sentiment score, green if positive, red if negative
      Right axis (line): closing price

    Args:
        df: merged dataframe with date, sentiment_score, close

    Returns:
        matplotlib Figure object (Streamlit can render this directly)
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # --- Sentiment bars (left axis) ---
    colors = ["#00C9A7" if s >= 0 else "#F31C96" for s in df["sentiment_score"]]
    ax1.bar(
        df["date"],
        df["sentiment_score"],
        color=colors,
        alpha=0.6,
        width=0.8,
        label="Sentiment score"
    )
    ax1.set_ylabel("Sentiment score", color="#EDEDED")
    ax1.axhline(y=0, color="#555555", linewidth=0.8, linestyle="--")
    ax1.tick_params(axis="y", labelcolor="#EDEDED")

    # Format x-axis dates nicely
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # --- Price line (right axis) ---
    ax2 = ax1.twinx()    # twinx() creates a second y-axis sharing the same x-axis
    ax2.plot(
        df["date"],
        df["close"],
        color="#4E3788",
        linewidth=2,
        label="Close price"
    )
    ax2.set_ylabel("Close price", color="#4E3788")
    ax2.tick_params(axis="y", labelcolor="#E8EAF0" )

    # --- Styling ---
    fig.patch.set_facecolor("#0F1F3D")
    ax1.set_facecolor("#0A1628")
    ax1.tick_params(axis="x", colors="#E8EAF0")
    ax1.tick_params(axis="y", labelcolor="#E8EAF0")
    ax1.set_ylabel("Sentiment score", color="#E8EAF0")
    ax1.spines[["top", "right", "left", "bottom"]].set_color("#1E3A5F")
    ax2.spines[["top", "right", "left", "bottom"]].set_color("#1E3A5F")
    ax2.tick_params(axis="y", labelcolor="#E8EAF0")
    ax2.set_ylabel("Close price", color="#4E3788")

    fig.tight_layout()

    return fig