# stock_data_fetcher.py
from datetime import datetime
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class StockDataFetcher:
    def __init__(self, api_connection, start_date, end_date):
        """
        Initialize the data fetcher to retrieve historical data from Alpaca's Historical Data API.
        """
        self.api_connection = api_connection
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self, tickers):
        """
        Fetch stock data for multiple tickers using Alpaca's Historical Data API.
        """
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=datetime.strptime(self.start_date, '%Y-%m-%d'),
            end=datetime.strptime(self.end_date, '%Y-%m-%d')
        )
        bars = self.api_connection.historical_data_client.get_stock_bars(request_params)

        # Reset index to get 'symbol' as a column (if needed)
        df = bars.df.reset_index()  # Ensure 'symbol' is a column

        # Debugging: Print the first few rows to inspect
        print(df.head())
        print(df.columns)

        # Return the DataFrame with stock data for multiple tickers
        return df
