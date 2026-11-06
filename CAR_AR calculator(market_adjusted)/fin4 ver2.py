import yfinance as yf
import pandas as pd
import numpy as np
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

#Calculates Abnormal Returns, and Cumulative Abnormal Returns
daily_returns = price_data.pct_change(fill_method=None).dropna(how='all')
market_returns = daily_returns['^GSPC']
abnormal_returns = daily_returns[sample_tickers].sub(market_returns, axis=0)

post_event_ar = abnormal_returns.loc[event_date:] [sample_tickers]
cumulative_ar = (1 + post_event_ar).cumprod() - 1

#A function to calculate average the AR, and CAR stats, and provide corresponding t-stats
def show_stats(df, tickers):
    df_stats = df.copy() 

    df_stats['Average'] = df_stats[tickers].mean(axis=1)
    df_stats['Median'] = df_stats[tickers].median(axis=1)
    df_stats['Min'] = df_stats[tickers].min(axis=1)
    df_stats['Max'] = df_stats[tickers].max(axis=1)
    df_stats['Std Dev'] = df_stats[tickers].std(axis=1)

    N = len(tickers)

    df_stats['T-Stat'] = df_stats.apply(
        lambda row: row['Average'] / (row['Std Dev'] / np.sqrt(N)) if row['Std Dev'] > 0 else 0, 
        axis=1
    )

    return df_stats

finalized_ar = show_stats(abnormal_returns, sample_tickers)
finalized_car = show_stats(cumulative_ar, sample_tickers)

#Export, outputs data to a .csv file in the parent directory of this python script
finalized_ar.to_csv(Path(__file__).parent / f"market_adjusted_ar_{obs_start}_to_{obs_end}.csv")
finalized_car.to_csv(Path(__file__).parent / f"cumulative_ar_{obs_start}_to_{obs_end}.csv")