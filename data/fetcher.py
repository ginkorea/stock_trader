# stock_data_fetcher.py
from datetime import datetime
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


# stock_data_fetcher.py
class StockDataFetcher:
    def __init__(self, api_connection, start_date, end_date):
        """
        Initialize the data fetcher to retrieve historical data from Alpaca's Historical Data API.
        """
        self.api_connection = api_connection
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self, ticker):
        """
        Fetch stock data for a specific ticker using Alpaca's Historical Data API.
        Include Open, High, Low, Close, and Volume.
        """
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=datetime.strptime(self.start_date, '%Y-%m-%d'),
            end=datetime.strptime(self.end_date, '%Y-%m-%d')
        )
        bars = self.api_connection.historical_data_client.get_stock_bars(request_params)
        return bars.df[['open', 'high', 'low', 'close', 'volume']]  # Return OHLCV data


    def fetch_multiple(self, tickers):
        """
        Fetch stock data for multiple tickers.

        :param tickers: A list of stock ticker symbols.
        :return: Dictionary with ticker symbols as keys and DataFrames as values.
        """
        all_data = {}
        for ticker in tickers:
            all_data[ticker] = self.fetch_data(ticker)
        return all_data
