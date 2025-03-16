import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

def get_stock_data(ticker="AAPL", period="5y", interval="1d"):
    """
    get the historical data of the stock
    ticker: stock code, e.g. 'AAPL' for Apple
    period: time range, e.g. '5y' for 5 years
    interval: time interval, e.g. '1d' for daily data
    """
    # download data
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    # reset the index, make Date as a column
    df = df.reset_index()
    
    return df

# get the 5-year daily data of Apple
# apple_data = get_stock_data("AAPL", "5y", "1d")
# print(apple_data.head())

# get the data of multiple stocks
# tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
# stock_data = {}

# for ticker in tickers:
#     stock_data[ticker] = get_stock_data(ticker, "3y", "1d")

def save_for_lstm(ticker="AAPL", period="5y", interval="1d"):
    """
    get stock data and save it as the format required by the LSTM model
    
    parameters:
    ticker: stock code, e.g. 'AAPL' for Apple
    period: time range, e.g. '5y' for 5 years
    interval: time interval, e.g. '1d' for daily data
    filename: name of the saved file
    """
    # get data
    df = get_stock_data(ticker, period, interval)
    
    # ensure the data contains the required columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            if col == 'Date' and 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'Date'}, inplace=True)
            else:
                raise ValueError(f"the data is missing the required column: {col}")
    
    # ensure the data is sorted by date
    df = df.sort_values('Date')
    
    # handle missing values
    df = df.fillna(method='ffill')  # fill the missing values with the last valid value
    
    # save as CSV
    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv(os.path.join('data', f"{ticker}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
    print(f"data saved to data/{ticker}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv, ready for LSTM model")
    
    # show the statistical information of the data
    print(f"\nstatistical information:")
    print(f"time range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"number of data: {len(df)}")
    print(f"closing price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
    
    return df

# save the data of Apple for the LSTM model
# lstm_data = save_for_lstm("AAPL", "5y", "1d", "stock_prices.csv")

# if you want to save the data of other stocks, uncomment the following lines and modify the parameters
# lstm_data = save_for_lstm("MSFT", "5y", "1d", "stock_prices.csv")  # Microsoft
# lstm_data = save_for_lstm("GOOG", "5y", "1d", "stock_prices.csv")  # Google
# lstm_data = save_for_lstm("AMZN", "5y", "1d", "stock_prices.csv")  # Amazon
# lstm_data = save_for_lstm("BABA", "5y", "1d", "stock_prices.csv")  # Alibaba
# lstm_data = save_for_lstm("0700.HK", "5y", "1d", "stock_prices.csv")  # Tencent



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get stock data')
    
    parser.add_argument('--ticker', type=str, default='GOOG', help='stock code')
    parser.add_argument('--period', type=str, default='5y', help='time range')
    parser.add_argument('--interval', type=str, default='1d', help='time interval')
    args = parser.parse_args()
    
    save_for_lstm(args.ticker, args.period, args.interval)
