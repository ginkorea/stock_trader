# stock_data_fetcher.py
from datetime import datetime
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd


class StockDataFetcher:
    def __init__(self, api_connection, start_date, end_date, verbose=False):
        """
        Initialize the data fetcher to retrieve historical data from Alpaca's Historical Data API.
        """
        self.api_connection = api_connection
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose

    def fetch_data(self, tickers):
        """
        Fetch stock data for one or more tickers using Alpaca's Historical Data API.
        If a single ticker is provided, fetch data for that ticker alone.
        """
        all_data = []

        # Handle multiple tickers or single ticker
        tickers = [tickers] if isinstance(tickers, str) else tickers

        for ticker in tickers:
            # Create a request for a single ticker
            request_params = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=datetime.strptime(self.start_date, '%Y-%m-%d'),
                end=datetime.strptime(self.end_date, '%Y-%m-%d')
            )

            bars = self.api_connection.historical_data_client.get_stock_bars(request_params)

            # Ensure data for the ticker is not empty
            if bars.df.empty:
                print(f"No data available for ticker {ticker}. Skipping.")
                continue

            # Reset index to ensure 'symbol' is a column
            df = bars.df.reset_index()

            # Debugging: Print the first few rows to inspect
            if self.verbose:
                print(f"Data for ticker {ticker}:")
                print(df.head())
                print(df.columns)

            all_data.append(df)

        # Combine all fetched data into a single DataFrame
        if len(all_data) == 0:
            raise ValueError("No valid stock data fetched for any of the tickers.")
        full_df = pd.concat(all_data)

        # Return the combined DataFrame
        return full_df
