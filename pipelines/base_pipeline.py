from abc import ABC, abstractmethod
from data.fetcher import StockDataFetcher
from connection.client import APIConnection


class BasePipeline(ABC):
    def __init__(self, start_date, end_date, window_size, pred_days, ticker=None):
        """
        Base Pipeline class for shared functionality.

        Args:
            start_date (str): Start date for stock data.
            end_date (str): End date for stock data.
            window_size (int): Number of days to consider for training.
            pred_days (int): Days ahead to predict.
            ticker (str or list): Ticker symbol or list of ticker symbols.
        """
        self.api_connection = APIConnection()

        # Handle both a single ticker or a list of tickers
        if isinstance(ticker, str):
            self.tickers = [ticker]
        elif isinstance(ticker, list):
            self.tickers = ticker
        else:
            raise ValueError("Ticker must be a string (single ticker) or a list of tickers.")

        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.pred_days = pred_days

    def fetch_and_preprocess_data(self, processor_class):
        """
        Fetches and preprocesses stock data for the provided tickers using the processor class.

        Args:
            processor_class: Processor class to handle the data preprocessing.

        Returns:
            Preprocessed data.
        """
        # Fetch stock data
        data_fetcher = StockDataFetcher(self.api_connection, start_date=self.start_date, end_date=self.end_date)
        stock_data = data_fetcher.fetch_data(self.tickers)

        # Check if data is empty or incomplete
        if stock_data.empty:
            raise ValueError(
                f"Fetched data for tickers {self.tickers} is empty. Please check the stock symbol(s) or date range."
            )

        # Preprocess the data
        data_processor = processor_class(window_size=self.window_size, pred_days=self.pred_days)
        preprocessed_data = data_processor.preprocess(stock_data, self.tickers)

        if not preprocessed_data:
            raise ValueError("Preprocessed data is empty. Check if there are valid trading days in the given range.")

        print("Data preprocessing complete.")
        return preprocessed_data

    @abstractmethod
    def train_model(self, *args, **kwargs):
        """
        Abstract method for training the model.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Abstract method for making predictions.
        """
        pass
