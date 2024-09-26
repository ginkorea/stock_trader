import pandas as pd
from datetime import datetime
import pytz
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest, StockTradesRequest

class Ticker:
    def __init__(self, symbol: str, api_connection):
        self.symbol = symbol
        self.api_connection = api_connection

    @staticmethod
    def utc_to_eastern(utc_dt):
        """Convert UTC to Eastern time."""
        eastern_tz = pytz.timezone('US/Eastern')
        return utc_dt.astimezone(eastern_tz)

    def is_trading_hour(self, utc_timestamp):
        """Check if a timestamp is within trading hours."""
        et_timestamp = self.utc_to_eastern(utc_timestamp)
        return datetime(1900, 1, 1, 9, 30).time() <= et_timestamp.time() <= datetime(1900, 1, 1, 16, 0).time()

    def fetch_stock_data(self, start_date: str, end_date: str, timeframe=TimeFrame.Minute) -> pd.DataFrame:
        """Fetch minute-level stock data (bars) for the given symbol."""
        start_date_formatted = f"{start_date}T00:00:00Z"
        end_date_formatted = f"{end_date}T23:59:59Z"

        bars_request = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=timeframe,
            start=start_date_formatted,
            end=end_date_formatted
        )

        bars = self.api_connection.historical_data_client.get_stock_bars(bars_request)
        df = bars.df.reset_index()

        # Convert timestamps to Eastern time and check trading hours
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).apply(self.utc_to_eastern)
            df['is_trading'] = df['timestamp'].apply(self.is_trading_hour)
        return df

    def fetch_high_resolution_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch high-resolution (trades) data for a given symbol."""
        trades_request = StockTradesRequest(
            symbol_or_symbols=self.symbol,
            start=start_date,
            end=end_date
        )
        trades = self.api_connection.historical_data_client.get_stock_trades(trades_request)

        high_res_data = pd.DataFrame([{
            "timestamp": trade.timestamp,
            "price": trade.price,
            "size": trade.size,
            "exchange": trade.exchange,
            "conditions": trade.conditions,
            "tape": trade.tape
        } for trade in trades.df[self.symbol]])

        return high_res_data
