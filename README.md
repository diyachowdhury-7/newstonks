# newstonks 📰📈

**Live Demo**: https://newstonks-cwdfif8nmxvpqienfk6iuu.streamlit.app/

A news sentiment analysis web app that scrapes headlines for any stock ticker, classifies them as positive/negative/neutral using a fine-tuned NLP model, and plots sentiment against price movement over time with a lag correlation analysis to test whether yesterday's news predicts today's returns.

You enter a ticker and date range: the app fetches real headlines from Alpha Vantage, runs each one through FinBERT, aggregates daily sentiment scores, and overlays them on the stock's closing price. A second tab runs a statistical lag analysis to see if sentiment carries predictive signal.

Unlike generic sentiment tools trained on tweets or reviews, this uses FinBERT, a BERT model fine-tuned specifically on financial news, which understands domain language like "missed estimates", "guidance raised", and "regulatory probe" the way a finance analyst would.

## How it works

**Data pipeline**
- Headlines fetched from Alpha Vantage News Sentiment API, filtered by ticker and date range
- Weekend sentiment (Saturday + Sunday) accumulated and rolled into the following Monday's score to align with market trading days
- Price data fetched via yfinance (OHLCV), daily percentage return computed as the target variable

**Sentiment classification**
- Each headline scored by `ProsusAI/finbert` via HuggingFace `pipeline()`
- Positive → +confidence score, Negative → -confidence score, Neutral → 0
- All headlines per day averaged into a single daily sentiment score

**Lag correlation analysis**
- Pearson correlation computed between sentiment and price return at offsets 0–5 days
- p-value reported alongside each coefficient — correlations with p ≥ 0.05 are flagged as statistically insignificant regardless of r value
- Answers the question: does news from N days ago predict today's return?

**Visualization**
- Tab 1: dual-axis chart — sentiment bars (green/red) overlaid with closing price line (purple)
- Tab 2: horizontal bar chart of correlation vs lag, color-coded by statistical significance

## Project structure

```
newstonks/
├── app.py
├── .env
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── src/
│   ├── __init__.py
│   ├── fetch_news.py
│   ├── sentiment.py
│   ├── price.py
│   ├── merge.py
│   └── analysis.py
└── components/
    ├── __init__.py
    ├── chart_overlay.py
    └── chart_lag.py
```

## Stack

| Layer | Tool |
|---|---|
| Sentiment model | ProsusAI/FinBERT via HuggingFace Transformers |
| News data | Alpha Vantage News Sentiment API |
| Price data | yfinance |
| Statistical analysis | scipy (Pearson r + p-value) |
| Data processing | pandas |
| Visualization | matplotlib |
| Web app | Streamlit |
| Deployment | Streamlit Community Cloud |

## Input

Enter any valid ticker in the sidebar. Examples:
- US stocks: `AAPL`, `TSLA`, `NVDA`, `MSFT`
- Indian stocks: `RELIANCE.NS`, `INFY.NS`, `TCS.NS`

Select a date range (up to 2 years back on free Alpha Vantage tier).

## What I learned

- How transformer-based models differ from classical NLP, why FinBERT outperforms TF-IDF or VADER on financial text
- Signed confidence scoring as a way to encode both polarity and certainty into a single numeric signal
- Why percentage return (not raw price) is the correct variable to correlate against sentiment, raw prices are non-stationary and trend over time
- Weekend data alignment as a real data engineering problem in financial NLP
- The difference between correlation magnitude and statistical significance, a high r with p > 0.05 is meaningless with small samples
- Time-series lag analysis as a method to test predictive signal vs contemporaneous correlation
- Caching heavy model loads in Streamlit with `@st.cache_resource` to avoid reloading 400MB on every rerun
