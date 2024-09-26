from portfolio.ticker import Ticker
from typing import Optional
from alpaca.data.timeframe import TimeFrame

class Portfolio:
    def __init__(self, api_connection):
        self.api_connection = api_connection
        self.tickers = {}

    def add_ticker(self, symbol: str):
        """Add a new ticker to the portfolio."""
        if symbol not in self.tickers:
            self.tickers[symbol] = Ticker(symbol, self.api_connection)

    def remove_ticker(self, symbol: str):
        """Remove a ticker from the portfolio."""
        if symbol in self.tickers:
            del self.tickers[symbol]

    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Retrieve a ticker from the portfolio."""
        return self.tickers.get(symbol, None)

    def fetch_data_for_all(self, start_date: str, end_date: str, timeframe=TimeFrame.Minute):
        """Fetch stock data for all tickers in the portfolio."""
        data = {}
        for symbol, ticker in self.tickers.items():
            data[symbol] = ticker.fetch_stock_data(start_date, end_date, timeframe)
        return data
