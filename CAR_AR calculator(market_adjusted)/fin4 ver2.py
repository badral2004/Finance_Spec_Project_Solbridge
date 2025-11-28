import yfinance as yf
import pandas as pd
from pathlib import Path

#CONFIG
ticker_input = Path(__file__).parent / "sample_input.csv"
benchmark = '^GSPC'
obs_start = "2025-01-15"
obs_end = "2025-01-24"
event_date = "2025-01-21" #The calculation of Cumulative Abnormal Returns will start from this date.

#This part of the code reads the tickers from the CSV, and adds the benchmark on top
df_ticker = pd.read_csv(ticker_input)
sample_tickers = df_ticker['ticker'].tolist()
pull_list = sample_tickers + [benchmark]

#This part pulls historical prices for the given ticker list
price_data = yf.download(pull_list, start=obs_start, end=obs_end, auto_adjust=False)['Adj Close']

#Calculates Abnormal Returns
daily_returns = price_data.pct_change(fill_method=None).dropna(how='all')
market_returns = daily_returns['^GSPC']
abnormal_returns = daily_returns[sample_tickers].sub(market_returns, axis=0)
abnormal_returns['Average'] = abnormal_returns.mean(axis=1)

#This part calculates Cumulative Abnormal Returns
post_event_ar = abnormal_returns.loc[event_date:]  # Slice from t=0 onward
cumulative_ar = (1 + post_event_ar).cumprod() - 1
cumulative_ar['Average'] = cumulative_ar.mean(axis=1)

#outputs AR, and cAR
abnormal_returns.to_csv(Path(__file__).parent / "market_adjusted_ar.csv")
cumulative_ar.to_csv(Path(__file__).parent / "cumulative_AR.csv")