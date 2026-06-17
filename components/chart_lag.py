import pandas as pd
import matplotlib.pyplot as plt


def plot_lag_correlation(lag_df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart showing correlation at each lag.
    Significant bars (p < 0.05) are solid purple.
    Non-significant bars are muted gray.

    Args:
        lag_df: output from compute_lag_correlations()

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ["#4E3788" if sig else "#243650" for sig in lag_df["significant"]]

    bars = ax.barh(
        lag_df["lag_label"],
        lag_df["correlation"],
        color=colors,
        alpha=0.85,
        height=0.5
    )

    # Add p-value labels on each bar
    for bar, (_, row) in zip(bars, lag_df.iterrows()):
        x_pos = bar.get_width()
        label = f"r={row['correlation']},  p={row['p_value']}"
    
    # If bar goes left (negative), put label on the right side of zero
    # If bar goes right (positive), put label to the right of bar
        if x_pos < 0:
            ax.text(
                0.02,                          # always place to the right of zero line
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=9,
                color="#E8EAF0"
            )
        else:
            ax.text(
                x_pos + 0.02,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=9,
                color="#E8EAF0"
            )

    # Vertical line at r=0
    ax.axvline(x=0, color="#888888", linewidth=1, linestyle="--")

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#7C6FF7", label="Significant (p < 0.05)"),
        Patch(facecolor="#555555", label="Not significant (p ≥ 0.05)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", 
              facecolor="#1A1A1A", labelcolor="#EDEDED", fontsize=9)

    ax.set_xlabel("Pearson r", color="#EDEDED")
    ax.set_xlim(-1, 1.3)   # extra space on right for labels

    fig.patch.set_facecolor("#1A2E45")
    ax.set_facecolor("#121F30")
    ax.tick_params(colors="#EDEDED")
    ax.spines[["top", "right", "left", "bottom"]].set_color("#333333")

    fig.tight_layout()
    return fig