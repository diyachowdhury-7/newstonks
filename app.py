import streamlit as st
import pandas as pd
from datetime import date, timedelta

from src.fetch_news import fetch_headlines
from src.sentiment import score_all_headlines
from src.price import fetch_price_data
from src.merge import merge_sentiment_and_price
from src.analysis import compute_lag_correlations
from components.chart_overlay import plot_sentiment_vs_price
from components.chart_lag import plot_lag_correlation

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme toggle + custom CSS ─────────────────────────────────────────────────
if "light_mode" not in st.session_state:
    st.session_state.light_mode = False

# Inject CSS based on mode
if st.session_state.light_mode:
    bg         = "#F4F6FB"
    bg2        = "#FFFFFF"
    text       = "#1A1A2E"
    subtext    = "#555577"
    card_bg    = "#FFFFFF"
    border     = "#D0D4E8"
    accent1    = "#4E3788"
    accent2    = "#F31C96"
    accent3    = "#166581"
else:
    bg         = "#121F30"
    bg2        = "#1A2E45"
    text       = "#E8EAF0"
    subtext    = "#8A9BB5"
    card_bg    = "#1A2E45"
    border     = "#243650"
    accent1    = "#4E3788"
    accent2    = "#F31C96"
    accent3    = "#166581"

st.markdown(f"""
<style>
    /* Page background */
    .stApp {{
        background-color: {bg};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {bg2};
        border-right: 1px solid {border};
    }}

    /* Cards */
    .sentiment-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }}

    /* Metric cards */
    div[data-testid="metric-container"] {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 16px 20px;
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
        border-bottom: 1px solid {border};
        padding-bottom: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px 8px 0 0;
        color: {subtext};
        padding: 8px 20px;
        font-size: 14px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {accent1} !important;
        color: white !important;
    }}

    /* Chart containers */
    .chart-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 24px;
        margin-top: 16px;
    }}

    /* General text */
    .stMarkdown, p, span, label {{
        color: {text} !important;
    }}

    /* Headings */
    h1, h2, h3 {{
        color: {text} !important;
        font-weight: 600 !important;
    }}

    /* Button */
    .stButton > button {{
        background: linear-gradient(135deg, {accent1}, {accent2});
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }}
    .stButton > button:hover {{
        opacity: 0.85;
    }}

    /* Input fields */
    .stTextInput > div > div > input,
    .stDateInput > div > div > input {{
        background: {bg};
        border: 1px solid {border};
        border-radius: 8px;
        color: {text};
    }}

    /* Hide streamlit branding */
    #MainMenu, footer {{ visibility: hidden; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {bg}; }}
    ::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ── Header row ────────────────────────────────────────────────────────────────
col_title, col_toggle = st.columns([6, 1])
with col_title:
    st.markdown(f"<h1 style='margin-bottom:0'>📈 Stock Sentiment Analyzer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{subtext}; margin-top:4px'>News sentiment vs price movement — FinBERT + Finnhub</p>", unsafe_allow_html=True)

with col_toggle:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("☀️ Light" if not st.session_state.light_mode else "🌙 Dark"):
        st.session_state.light_mode = not st.session_state.light_mode
        st.rerun()

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{accent2}'>⚙️ Settings</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    ticker = st.text_input(
        "Ticker symbol",
        value="AAPL",
        help="US stocks: AAPL, TSLA | Indian: RELIANCE.NS, INFY.NS"
    ).upper().strip()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=date.today() - timedelta(days=90))
    with col2:
        end_date = st.date_input("End", value=date.today())

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("Analyze", type="primary", use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:12px; color:{subtext}; line-height:2'>
        📰 Data: Finnhub News API<br>
        🤖 Model: ProsusAI/FinBERT<br>
        💹 Prices: yfinance<br>
        📅 Weekend sentiment rolled into Monday
    </div>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
if not analyze:
    st.markdown(f"""
    <div class='sentiment-card' style='text-align:center; padding: 60px 40px;'>
        <h2 style='color:{accent2}'>Get Started</h2>
        <p style='color:{subtext}; font-size:16px'>
            Enter a ticker symbol and date range in the sidebar,<br>
            then click <b>Analyze</b> to see sentiment vs price movement.
        </p>
        <br>
        <p style='color:{subtext}; font-size:13px'>
            Try: AAPL · TSLA · RELIANCE.NS · INFY.NS · MSFT
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

start_str = start_date.strftime("%Y-%m-%d")
end_str   = end_date.strftime("%Y-%m-%d")

try:
    with st.spinner("📰 Fetching headlines from Finnhub..."):
        headlines_df = fetch_headlines(ticker, start_str, end_str)

    with st.spinner(f"🤖 Running FinBERT on {len(headlines_df)} headlines..."):
        sentiment_df = score_all_headlines(headlines_df)

    with st.spinner("💹 Fetching price data..."):
        price_df = fetch_price_data(ticker, start_str, end_str)

    merged_df = merge_sentiment_and_price(sentiment_df, price_df)

    if merged_df.empty:
        st.error("No overlapping data found. Try a wider date range.")
        st.stop()

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    price_change = (
        (merged_df["close"].iloc[-1] - merged_df["close"].iloc[0])
        / merged_df["close"].iloc[0] * 100
    )
    avg_sent = merged_df["sentiment_score"].mean()
    sent_label = "🟢 Positive" if avg_sent > 0.05 else ("🔴 Negative" if avg_sent < -0.05 else "⚪ Neutral")

    m1.metric("Headlines scraped", f"{len(headlines_df):,}")
    m2.metric("Trading days analyzed", len(merged_df))
    m3.metric("Avg sentiment", f"{avg_sent:.3f}", sent_label)
    m4.metric("Price change", f"{price_change:+.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊  Sentiment vs Price", "🔬  Lag Analysis"])

    with tab1:
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{ticker} — Daily Sentiment vs Closing Price</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{subtext}; font-size:13px'>Green bars = positive sentiment · Red bars = negative · Purple line = closing price</p>", unsafe_allow_html=True)

        fig1 = plot_sentiment_vs_price(merged_df)
        st.pyplot(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 View raw data table"):
            st.dataframe(
                merged_df[["date", "sentiment_score", "close", "pct_change"]].rename(columns={
                    "sentiment_score": "Sentiment",
                    "close": "Close price",
                    "pct_change": "Daily return (%)"
                }),
                use_container_width=True
            )

    with tab2:
        st.markdown(f"<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Lag Correlation Analysis</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{subtext}; font-size:13px'>Does past sentiment predict future returns? Purple = statistically significant (p &lt; 0.05) · Gray = not significant</p>", unsafe_allow_html=True)

        lag_df = compute_lag_correlations(merged_df)
        fig2   = plot_lag_correlation(lag_df)
        st.pyplot(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        best      = lag_df.loc[lag_df["correlation"].abs().idxmax()]
        sig_text  = "statistically significant (p < 0.05)" if best["significant"] else "not statistically significant"
        direction = "positive" if best["correlation"] > 0 else "negative"

        st.info(
            f"**Strongest signal at Lag {int(best['lag'])}d:** "
            f"r = {best['correlation']}, p = {best['p_value']} — {sig_text}. "
            f"{'Sentiment may weakly predict price movement ' + str(int(best['lag'])) + ' day(s) ahead.' if best['significant'] else 'No reliable predictive relationship found in this date range.'}"
        )

        with st.expander("📋 View lag table"):
            st.dataframe(lag_df, use_container_width=True)

except ValueError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.exception(e)