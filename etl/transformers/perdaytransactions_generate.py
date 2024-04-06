if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import datetime
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from pathlib import Path

def create_market_cal(start, end):
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start, end)
    market_cal = mcal.date_range(schedule, frequency='1D')
    market_cal = market_cal.tz_localize(None)
    market_cal = [i.replace(hour=0) for i in market_cal]
    return market_cal

def get_data(stocks, start, end):
    def data(ticker):
        print(f'Getting data for {ticker}...')
        df = yf.download(ticker, start=start, end=(datetime.datetime.strptime(end, "%Y-%m-%d") + datetime.timedelta(days=1)))
        df['symbol'] = ticker
        df.index = pd.to_datetime(df.index)
        return df
    datas = map(data,stocks)
    return(pd.concat(datas, keys=stocks, names=['ticker', 'date'], sort=True))

def position_adjust(daily_positions, sale):
  stocks_with_sales = pd.DataFrame(columns=["closed_stock_gain/(loss)"])
  buys_before_start = daily_positions[daily_positions['transaction_type'] == 'Buy'].sort_values(by='open_date')
  for position in buys_before_start[buys_before_start['symbol'] == sale[1]['symbol']].iterrows():
      sale[1]['adj_cost'] = pd.to_numeric(sale[1]["adj_cost"], errors='coerce')
      sale[1]['adj_cost_per_share'] = pd.to_numeric(sale[1]["adj_cost_per_share"], errors='coerce')
      position[1]['adj_cost_per_share'] = pd.to_numeric(position[1]["adj_cost_per_share"], errors='coerce')
      sale[1]['qty'] = pd.to_numeric(sale[1]["qty"], errors='coerce')
      if (position[1]['qty'] <= sale[1]['qty']) & (sale[1]['qty'] > 0):
          position[1]["closed_stock_gain/(loss)"] += (sale[1]['adj_cost_per_share'] - position[1]['adj_cost_per_share']) * position[1]['qty']
          sale[1]['qty'] -= position[1]['qty']
          position[1]['qty'] = 0
      elif (position[1]['qty'] > sale[1]['qty']) & (sale[1]['qty'] > 0):
          position[1]["closed_stock_gain/(loss)"] += (sale[1]['adj_cost_per_share'] - position[1]['adj_cost_per_share']) * sale[1]['qty']
          position[1]['qty'] -= sale[1]['qty']
          sale[1]['qty'] -= sale[1]['qty']
      stocks_with_sales = stocks_with_sales._append(position[1])
  return stocks_with_sales

def portfolio_start_balance(portfolio, start_date):
  positions_before_start = portfolio[portfolio['open_date'] <= datetime.datetime.strptime(start_date, "%Y-%m-%d").date()]
  future_positions = portfolio[portfolio['open_date'] >= datetime.datetime.strptime(start_date, "%Y-%m-%d").date()]
  sales = positions_before_start[positions_before_start['transaction_type'] == 'Sell.FIFO'].groupby(['symbol'])['qty'].sum()
  sales = sales.reset_index()
  positions_no_change = positions_before_start[~positions_before_start['symbol'].isin(sales['symbol'].unique())]
  adj_positions_df = pd.DataFrame()
  for sale in sales.iterrows():
      adj_positions = position_adjust(positions_before_start, sale)
      adj_positions_df = adj_positions_df.append(adj_positions)
  adj_positions_df = adj_positions_df._append(positions_no_change)
  adj_positions_df = adj_positions_df._append(future_positions)
  adj_positions_df = adj_positions_df[adj_positions_df['qty'] > 0]
  return adj_positions_df

def fifo(daily_positions, sales, date):
  sales = sales[sales['open_date'] == date]
  daily_positions = daily_positions[daily_positions['open_date'] <= date]
  positions_no_change = daily_positions[~daily_positions['symbol'].isin(sales['symbol'].unique())]
  adj_positions = pd.DataFrame()
  for sale in sales.iterrows():
      adj_positions = adj_positions._append(position_adjust(daily_positions, sale))
  adj_positions = adj_positions._append(positions_no_change)
  return adj_positions

def time_fill(portfolio, market_cal, stocks_end):
  portfolio['closed_stock_gain/(loss)'] = 0
  sales = portfolio[portfolio['transaction_type'] == 'Sell.FIFO'].groupby(['symbol','open_date', 'adj_cost', 'adj_cost_per_share'])['qty'].sum()
  sales = sales.reset_index()
  sales['open_date'] = (sales['open_date'] + pd.DateOffset(days=1)).apply(lambda date: min(market_cal, key=lambda x: abs(x - date)))
  per_day_balance = []
  for date in market_cal:
      if (sales['open_date'] == date).any():
          future_txns = portfolio[(portfolio['open_date'] > date) & (portfolio['open_date'] <= datetime.datetime.strptime(stocks_end, "%Y-%m-%d").date())]
          portfolio = fifo(portfolio, sales, date)
          portfolio = portfolio._append(future_txns)
      daily_positions = portfolio[portfolio['open_date'] <= date]
      daily_positions = daily_positions[daily_positions['transaction_type'] == 'Buy']
      daily_positions['date_snapshot'] = date
      per_day_balance.append(daily_positions)
  return per_day_balance

def modified_cost_per_share(portfolio, adj_close, start_date):
  df = pd.merge(portfolio, adj_close, left_on=['date_snapshot', 'symbol'], right_on=['date', 'ticker'], how='left')
  df.rename(columns={'adj_close': 'symbol_adj_close'}, inplace=True)
  df['adj_cost_daily'] = df['symbol_adj_close'] * df['qty']
  df = df.drop(['ticker', 'date'], axis=1)
  return df

def portfolio_end_of_year_stats(portfolio, adj_close_end):
  adj_close_end = adj_close_end[adj_close_end['date'] == adj_close_end['date'].max()]
  portfolio_end_data = pd.merge(portfolio, adj_close_end, left_on='symbol', right_on='ticker')
  portfolio_end_data.rename(columns={'adj_close': 'ticker_end_date_close'}, inplace=True)
  portfolio_end_data = portfolio_end_data.drop(['ticker', 'date'], axis=1)
  return portfolio_end_data

def portfolio_start_of_year_stats(portfolio, adj_close_start):
  adj_close_start = adj_close_start[adj_close_start['date'] == adj_close_start['date'].min()]
  portfolio_start = pd.merge(portfolio, adj_close_start[['ticker', 'adj_close', 'date']], left_on='symbol', right_on='ticker')
  portfolio_start.rename(columns={'adj_close': 'ticker_start_date_close'}, inplace=True)
  portfolio_start['adj_cost_per_share'] = np.where(portfolio_start['open_date'] <= portfolio_start['date'], portfolio_start['ticker_start_date_close'], portfolio_start['adj_cost_per_share'])
  portfolio_start["adj_cost_per_share"] = pd.to_numeric(portfolio_start["adj_cost_per_share"], errors='coerce')
  portfolio_start['adj_cost'] = portfolio_start['adj_cost_per_share'] * portfolio_start['qty']
  portfolio_start = portfolio_start.drop(['ticker', 'date'], axis=1)
  return portfolio_start

def calc_returns(portfolio):
  portfolio['ticker_daily_return'] = portfolio.groupby('symbol')['symbol_adj_close'].pct_change()
  portfolio['ticker_return'] = portfolio['symbol_adj_close'] / portfolio['adj_cost_per_share'] - 1
  portfolio['ticker_share_value'] = portfolio['qty'] * portfolio['symbol_adj_close']
  portfolio['open_stock_gain/(loss)'] = portfolio['ticker_share_value'] - portfolio['adj_cost']
  portfolio = portfolio.dropna(axis=1, how='all') # Drops some columns that get added in by mistake
  # Add a column that is the current time the script is ran
  portfolio['python_script_last_run_time'] = datetime.datetime.now()
  return portfolio

def per_day_portfolio_calcs(per_day_holdings, daily_adj_close, stocks_start):
  df = pd.concat(per_day_holdings, sort=True)
  mcps = modified_cost_per_share(df, daily_adj_close, stocks_start)
  pes = portfolio_end_of_year_stats(mcps, daily_adj_close)
  pss = portfolio_start_of_year_stats(pes, daily_adj_close)
  returns = calc_returns(pss)
  return returns

# ROLLING SHARE VALUE METHOD
def format_returns(pdpc):
  # Sum of the ticker_share_value on each day
  ticker_Share_Value = pdpc.groupby(['date_snapshot'])[['ticker_share_value']].sum().reset_index()
  ticker_Share_Value = pd.melt(ticker_Share_Value, id_vars=['date_snapshot'],
                              value_vars=['ticker_share_value'])
  ticker_Share_Value.set_index('date_snapshot', inplace=True)
  ticker_Share_Value.rename(columns={'value': 'ticker_share_value'}, inplace=True)

  # Total ticker_share_value Weighted Return on each day
  grouped_metrics5 = pdpc.groupby(['date_snapshot', 'symbol'])['ticker_share_value'].sum().reset_index()
  grouped_metrics5 = grouped_metrics5.merge(ticker_Share_Value, on='date_snapshot', suffixes=('', '_total'))
  grouped_metrics5['weight'] = grouped_metrics5['ticker_share_value'] / grouped_metrics5['ticker_share_value_total']

  # Calculate daily returns for each symbol
  symbol_returns = pdpc.groupby(['date_snapshot', 'symbol'])['ticker_daily_return'].sum().reset_index()

  # Join the `grouped_metrics5` dataframe with the `symbol_returns` series
  grouped_metrics5 = pd.concat([grouped_metrics5, symbol_returns['ticker_daily_return']], axis=1, join='inner')

  # Calculate ticker weighted returns
  grouped_metrics5['ticker_weighted_return'] = grouped_metrics5['weight'] * grouped_metrics5['ticker_daily_return']

  # Group by date and calculate total weighted returns
  grouped_metrics5 = grouped_metrics5.groupby('date_snapshot')['ticker_weighted_return'].sum().reset_index()
  grouped_metrics5.rename(columns={'ticker_weighted_return': 'total_weighted_return'}, inplace=True)
  grouped_metrics5.set_index('date_snapshot', inplace=True)

  # calculate daily returns in % form
  grouped_metrics5['total_weighted_return'].fillna(0, inplace=True)
  grouped_metrics5['total_weighted_return'] = grouped_metrics5['total_weighted_return'].replace([np.inf, -np.inf], 0)
  grouped_metrics5['cumulative_total_weighted_return'] = (grouped_metrics5['total_weighted_return'].cumsum() * 100).ffill()

  return grouped_metrics5['total_weighted_return']

def assign_sectors(df):
  sectors_dict = {
        'Technology' : ['AAPL','ADBE', 'LRCX', 'MSFT','NVDA', 'VGT','CRM'],
        'Financials & Real Estate' : ['BLK', 'BN', 'SIVBQ', 'VFH','BAM','RLI','JPM','VNQ'],
        'Healthcare' : ['ABBV', 'MDT', 'TMO','ELV','VHT','MRK', 'OGN', 'UNH'],
        'Consumer Discretionary' : ['LULU','TJX','SBUX','VCR'],
        'Communications' : ['DIS', 'META', 'TMUS', 'GOOGL','VOX','ATVI'],
        'Industrials' : ['CAT','J','ROK','UNP','VIS'],
        'Consumer Staples' : ['WMT','VDC'],
        'Utilities' : ['AEP','NEE', 'SRE','UGI','VPU','AES','AWK'],
        'Materials' : ['VAW','SLGN'],
        'Energy' : ['VDE','EOG', 'XOM']
    }

  sectors = []
  for ticker in df['symbol']:
      found = False
      for sector, stocks in sectors_dict.items():
          if ticker in stocks:
              sectors.append(sector)
              found = True
              break

      if not found:
          sectors.append('None')

  df['sector'] = sectors
  return df

@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    print("Reading Portfolio Transactions from .CSV...")

    # Portfolio Inception date
    start_date='2020-04-30'

    # Read in the Portfolio Transactions
    portfolio_df = data
    portfolio_df['open_date'] = pd.to_datetime(portfolio_df['open_date']).dt.date

    # create a mask to filter the dataframe to only contain buy and sell orders
    mask = (portfolio_df['transaction_type'] == 'Buy') | (portfolio_df['transaction_type'] == 'Sell.FIFO')
    portfolio_df = portfolio_df[mask]

    # Create and Array of Unique tickers
    symbols = portfolio_df.symbol.unique()
    print(symbols)

    # Add Sector to each ticker
    portfolio_df = assign_sectors(portfolio_df)

    today = datetime.datetime.today()
    stocks_end = today.strftime("%Y-%m-%d")

    daily_adj_close = get_data(symbols, start_date, stocks_end)
    daily_adj_close.rename(columns = {'Adj Close':'adj_close'}, inplace = True) 
    daily_adj_close = daily_adj_close[['adj_close']].reset_index()

    market_cal = create_market_cal(start_date, stocks_end)

    print('Determining Active Portfolio...')

    active_portfolio = portfolio_start_balance(portfolio_df, start_date)

    print('Calculating Daily Position Snapshots...')

    positions_per_day = time_fill(active_portfolio, market_cal, stocks_end)

    pdpc = per_day_portfolio_calcs(positions_per_day, daily_adj_close, start_date)
    pdpc.drop(columns=['id'], inplace=True)

    return pdpc


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
