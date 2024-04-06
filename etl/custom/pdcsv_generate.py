if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import datetime
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

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
    return(pd.concat(datas, keys=stocks, names=['Ticker', 'Date'], sort=True))

def read_csv(csv_name):
    portfolio_df = pd.read_csv(f'{csv_name}.csv')
    portfolio_df['Open date'] = pd.to_datetime(portfolio_df['Open date'])
    return portfolio_df

def position_adjust(daily_positions, sale):
  stocks_with_sales = pd.DataFrame(columns=["Closed Stock Gain / (Loss)"])
  buys_before_start = daily_positions[daily_positions['Type'] == 'Buy'].sort_values(by='Open date')
  for position in buys_before_start[buys_before_start['Symbol'] == sale[1]['Symbol']].iterrows():
      sale[1]['Adj cost'] = pd.to_numeric(sale[1]["Adj cost"], errors='coerce')
      sale[1]['Adj cost per share'] = pd.to_numeric(sale[1]["Adj cost per share"], errors='coerce')
      position[1]['Adj cost per share'] = pd.to_numeric(position[1]["Adj cost per share"], errors='coerce')
      sale[1]['Qty'] = pd.to_numeric(sale[1]["Qty"], errors='coerce')
      if (position[1]['Qty'] <= sale[1]['Qty']) & (sale[1]['Qty'] > 0):
          position[1]["Closed Stock Gain / (Loss)"] += (sale[1]['Adj cost per share'] - position[1]['Adj cost per share']) * position[1]['Qty']
          sale[1]['Qty'] -= position[1]['Qty']
          position[1]['Qty'] = 0
      elif (position[1]['Qty'] > sale[1]['Qty']) & (sale[1]['Qty'] > 0):
          position[1]["Closed Stock Gain / (Loss)"] += (sale[1]['Adj cost per share'] - position[1]['Adj cost per share']) * sale[1]['Qty']
          position[1]['Qty'] -= sale[1]['Qty']
          sale[1]['Qty'] -= sale[1]['Qty']
      stocks_with_sales = stocks_with_sales._append(position[1])
  return stocks_with_sales

def portfolio_start_balance(portfolio, start_date):
  positions_before_start = portfolio[portfolio['Open date'] <= start_date]
  future_positions = portfolio[portfolio['Open date'] >= start_date]
  sales = positions_before_start[positions_before_start['Type'] == 'Sell.FIFO'].groupby(['Symbol'])['Qty'].sum()
  sales = sales.reset_index()
  positions_no_change = positions_before_start[~positions_before_start['Symbol'].isin(sales['Symbol'].unique())]
  adj_positions_df = pd.DataFrame()
  for sale in sales.iterrows():
      adj_positions = position_adjust(positions_before_start, sale)
      adj_positions_df = adj_positions_df.append(adj_positions)
  adj_positions_df = adj_positions_df._append(positions_no_change)
  adj_positions_df = adj_positions_df._append(future_positions)
  adj_positions_df = adj_positions_df[adj_positions_df['Qty'] > 0]
  return adj_positions_df

def fifo(daily_positions, sales, date):
  sales = sales[sales['Open date'] == date]
  daily_positions = daily_positions[daily_positions['Open date'] <= date]
  positions_no_change = daily_positions[~daily_positions['Symbol'].isin(sales['Symbol'].unique())]
  adj_positions = pd.DataFrame()
  for sale in sales.iterrows():
      adj_positions = adj_positions._append(position_adjust(daily_positions, sale))
  adj_positions = adj_positions._append(positions_no_change)
  return adj_positions

def time_fill(portfolio, market_cal, stocks_end):
  portfolio['Closed Stock Gain / (Loss)'] = 0
  sales = portfolio[portfolio['Type'] == 'Sell.FIFO'].groupby(['Symbol','Open date', 'Adj cost', 'Adj cost per share'])['Qty'].sum()
  sales = sales.reset_index()
  sales['Open date'] = (sales['Open date'] + pd.DateOffset(days=1)).apply(lambda date: min(market_cal, key=lambda x: abs(x - date)))
  per_day_balance = []
  for date in market_cal:
      if (sales['Open date'] == date).any():
          future_txns = portfolio[(portfolio['Open date'] > date) & (portfolio['Open date'] <= stocks_end)]
          portfolio = fifo(portfolio, sales, date)
          portfolio = portfolio._append(future_txns)
      daily_positions = portfolio[portfolio['Open date'] <= date]
      daily_positions = daily_positions[daily_positions['Type'] == 'Buy']
      daily_positions['Date Snapshot'] = date
      per_day_balance.append(daily_positions)
  return per_day_balance

def modified_cost_per_share(portfolio, adj_close, start_date):
  df = pd.merge(portfolio, adj_close, left_on=['Date Snapshot', 'Symbol'], right_on=['Date', 'Ticker'], how='left')
  df.rename(columns={'Adj Close': 'Symbol Adj Close'}, inplace=True)
  df['Adj cost daily'] = df['Symbol Adj Close'] * df['Qty']
  df = df.drop(['Ticker', 'Date'], axis=1)
  return df

def portfolio_end_of_year_stats(portfolio, adj_close_end):
  adj_close_end = adj_close_end[adj_close_end['Date'] == adj_close_end['Date'].max()]
  portfolio_end_data = pd.merge(portfolio, adj_close_end, left_on='Symbol', right_on='Ticker')
  portfolio_end_data.rename(columns={'Adj Close': 'Ticker End Date Close'}, inplace=True)
  portfolio_end_data = portfolio_end_data.drop(['Ticker', 'Date'], axis=1)
  return portfolio_end_data

def portfolio_start_of_year_stats(portfolio, adj_close_start):
  adj_close_start = adj_close_start[adj_close_start['Date'] == adj_close_start['Date'].min()]
  portfolio_start = pd.merge(portfolio, adj_close_start[['Ticker', 'Adj Close', 'Date']], left_on='Symbol', right_on='Ticker')
  portfolio_start.rename(columns={'Adj Close': 'Ticker Start Date Close'}, inplace=True)
  portfolio_start['Adj cost per share'] = np.where(portfolio_start['Open date'] <= portfolio_start['Date'], portfolio_start['Ticker Start Date Close'], portfolio_start['Adj cost per share'])
  portfolio_start["Adj cost per share"] = pd.to_numeric(portfolio_start["Adj cost per share"], errors='coerce')
  portfolio_start['Adj cost'] = portfolio_start['Adj cost per share'] * portfolio_start['Qty']
  portfolio_start = portfolio_start.drop(['Ticker', 'Date'], axis=1)
  return portfolio_start

def calc_returns(portfolio):
  portfolio['Ticker Daily Return'] = portfolio.groupby('Symbol')['Symbol Adj Close'].pct_change()
  portfolio['Ticker Return'] = portfolio['Symbol Adj Close'] / portfolio['Adj cost per share'] - 1
  portfolio['Ticker Share Value'] = portfolio['Qty'] * portfolio['Symbol Adj Close']
  portfolio['Open Stock Gain / (Loss)'] = portfolio['Ticker Share Value'] - portfolio['Adj cost']
  portfolio = portfolio.dropna(axis=1, how='all') # Drops some columns that get added in by mistake
  # Add a column that is the current time the script is ran
  portfolio['PythonScriptLastRunTime'] = datetime.datetime.now()
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
  # Sum of the ticker share value on each day
  Ticker_Share_Value = pdpc.groupby(['Date Snapshot'])[['Ticker Share Value']].sum().reset_index()
  Ticker_Share_Value = pd.melt(Ticker_Share_Value, id_vars=['Date Snapshot'],
                              value_vars=['Ticker Share Value'])
  Ticker_Share_Value.set_index('Date Snapshot', inplace=True)
  Ticker_Share_Value.rename(columns={'value': 'Ticker Share Value'}, inplace=True)

  # Total Ticker Share Value Weighted Return on each day
  grouped_metrics5 = pdpc.groupby(['Date Snapshot', 'Symbol'])['Ticker Share Value'].sum().reset_index()
  grouped_metrics5 = grouped_metrics5.merge(Ticker_Share_Value, on='Date Snapshot', suffixes=('', '_total'))
  grouped_metrics5['Weight'] = grouped_metrics5['Ticker Share Value'] / grouped_metrics5['Ticker Share Value_total']

  # Calculate daily returns for each symbol
  symbol_returns = pdpc.groupby(['Date Snapshot', 'Symbol'])['Ticker Daily Return'].sum().reset_index()

  # Join the `grouped_metrics5` dataframe with the `symbol_returns` series
  grouped_metrics5 = pd.concat([grouped_metrics5, symbol_returns['Ticker Daily Return']], axis=1, join='inner')

  # Calculate ticker weighted returns
  grouped_metrics5['Ticker Weighted Return'] = grouped_metrics5['Weight'] * grouped_metrics5['Ticker Daily Return']

  # display(grouped_metrics5)

  # Group by date and calculate total weighted returns
  grouped_metrics5 = grouped_metrics5.groupby('Date Snapshot')['Ticker Weighted Return'].sum().reset_index()
  grouped_metrics5.rename(columns={'Ticker Weighted Return': 'Total Weighted Return'}, inplace=True)
  grouped_metrics5.set_index('Date Snapshot', inplace=True)

  # calculate daily returns in % form
  grouped_metrics5['Total Weighted Return'].fillna(0, inplace=True)
  grouped_metrics5['Total Weighted Return'] = grouped_metrics5['Total Weighted Return'].replace([np.inf, -np.inf], 0)
  grouped_metrics5['Cumulative Total Weighted Return'] = (grouped_metrics5['Total Weighted Return'].cumsum() * 100).ffill()

  # display(grouped_metrics5)

  # Plot Data
  # line(grouped_metrics5, 'Total Weighted Return')
  # line(grouped_metrics5, 'Cumulative Total Weighted Return')

  return grouped_metrics5['Total Weighted Return']

# STATIC COST BASIS METHOD
# def format_returns(pdpc):
#   # Sum of the ticker share value on each day
#   Ticker_Share_Value = pdpc.groupby(['Date Snapshot'])[['Adj cost']].sum().reset_index()
#   Ticker_Share_Value = pd.melt(Ticker_Share_Value, id_vars=['Date Snapshot'],
#                               value_vars=['Adj cost'])
#   Ticker_Share_Value.set_index('Date Snapshot', inplace=True)
#   Ticker_Share_Value.rename(columns={'value': 'Adj cost'}, inplace=True)

#   # Total Ticker Share Value Weighted Return on each day
#   grouped_metrics5 = pdpc.groupby(['Date Snapshot', 'Symbol'])['Adj cost'].sum().reset_index()
#   grouped_metrics5 = grouped_metrics5.merge(Ticker_Share_Value, on='Date Snapshot', suffixes=('', '_total'))
#   grouped_metrics5['Weight'] = grouped_metrics5['Adj cost'] / grouped_metrics5['Adj cost_total']

#   # Calculate daily returns for each symbol
#   symbol_returns = pdpc.groupby(['Date Snapshot', 'Symbol'])['Ticker Daily Return'].sum().reset_index()

#   # Join the `grouped_metrics5` dataframe with the `symbol_returns` series
#   grouped_metrics5 = pd.concat([grouped_metrics5, symbol_returns['Ticker Daily Return']], axis=1, join='inner')

#   # Calculate ticker weighted returns
#   grouped_metrics5['Ticker Weighted Return'] = grouped_metrics5['Weight'] * grouped_metrics5['Ticker Daily Return']

#   # Group by date and calculate total weighted returns
#   grouped_metrics5 = grouped_metrics5.groupby('Date Snapshot')['Ticker Weighted Return'].sum().reset_index()
#   grouped_metrics5.rename(columns={'Ticker Weighted Return': 'Total Weighted Return'}, inplace=True)
#   grouped_metrics5.set_index('Date Snapshot', inplace=True)

#   # calculate daily returns in % form
#   grouped_metrics5['Total Weighted Return'].fillna(0, inplace=True)
#   grouped_metrics5['Total Weighted Return'] = grouped_metrics5['Total Weighted Return'].replace([np.inf, -np.inf], 0)
#   grouped_metrics5['Cumulative Total Weighted Return'] = (grouped_metrics5['Total Weighted Return'].cumsum() * 100).ffill()

#   # plot some test lines
#   line(grouped_metrics5, 'Total Weighted Return')
#   line(grouped_metrics5, 'Cumulative Total Weighted Return')

#   return grouped_metrics5['Total Weighted Return']

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
        # 'Real Estate' : ['VNQ']
        # 'Cash' : ['VMFXX']
    }

  sectors = []
  for ticker in df['Symbol']:
      found = False
      for sector, stocks in sectors_dict.items():
          if ticker in stocks:
              sectors.append(sector)
              found = True
              break

      if not found:
          sectors.append('None')

  df['Sector'] = sectors
  return df

def generate_report(filepath, start_date):
    print("Reading Portfolio Transactions from .CSV...")

    # Read in the Portfolio Transactions
    portfolio_df = read_csv(filepath)

    # create a mask to filter the dataframe to only contain buy and sell orders
    mask = (portfolio_df['Type'] == 'Buy') | (portfolio_df['Type'] == 'Sell.FIFO')

    portfolio_df = portfolio_df[mask]

    # Create and Array of Unique Tickers
    symbols = portfolio_df.Symbol.unique()
    print(symbols)

    # Add Sector to each Ticker
    portfolio_df = assign_sectors(portfolio_df)

    today = datetime.datetime.today()
    stocks_end = today.strftime("%Y-%m-%d")

    daily_adj_close = get_data(symbols, start_date, stocks_end)
    daily_adj_close = daily_adj_close[['Adj Close']].reset_index()

    market_cal = create_market_cal(start_date, stocks_end)

    print('Determining Active Portfolio...')

    active_portfolio = portfolio_start_balance(portfolio_df, start_date)

    print('Calculating Daily Position Snapshots...')

    positions_per_day = time_fill(active_portfolio, market_cal, stocks_end)

    pdpc = per_day_portfolio_calcs(positions_per_day, daily_adj_close, start_date)

    return pdpc

@custom
def transform_custom(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here

    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
