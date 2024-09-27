import numpy as np
from data.processor.base import StockDataProcessor

class StockDataRegressionProcessor(StockDataProcessor):
    def preprocess(self, stock_data, tickers):
        """
        Preprocess data for extracting open-to-high prices for linear regression.

        Args:
            stock_data (pd.DataFrame): DataFrame containing the stock data.
            tickers (list): List of ticker symbols.

        Returns:
            dict: Dictionary containing open-high data sequences for each ticker and time window.
        """
        # Extract features for each day and ticker
        all_features = self.extract_features(stock_data, tickers)

        # Extract open-high sequences for different time windows
        open_high_data = self.extract_open_high_sequences(all_features, tickers)

        return open_high_data

    def extract_open_high_sequences(self, all_features, tickers):
        """
        Extract open-to-high sequences for multiple time windows.

        Args:
            all_features (list): List of feature vectors for each day.
            tickers (list): List of ticker symbols.

        Returns:
            dict: Dictionary containing open-high data for each ticker and time window.
        """
        windows = [5, 15, 30, 90]
        open_high_data = {}

        for ticker_idx, ticker in enumerate(tickers):
            open_high_data[ticker] = {}
            for window in windows:
                X, y = self.get_open_high_for_window(all_features, window, ticker_idx)
                open_high_data[ticker][window] = {"X": X, "y": y}

        return open_high_data

    @staticmethod
    def get_open_high_for_window(all_features, window, ticker_idx):
        """
        Extract open and high prices for a given time window and ticker.

        Args:
            all_features (list): List of feature vectors for each day.
            window (int): Number of days to include in the window.
            ticker_idx (int): Index of the ticker in the feature vector.

        Returns:
            tuple: X (open prices), y (high prices) for the specified window.
        """
        X = []
        y = []

        for i in range(len(all_features) - window):
            open_prices = []
            high_prices = []

            for j in range(window):
                open_price = all_features[i + j][ticker_idx * 7]    # Open price
                high_price = all_features[i + j][ticker_idx * 7 + 1]  # High price
                open_prices.append(open_price)
                high_prices.append(high_price)

            X.append(open_prices)
            y.append(high_prices)

        return np.array(X), np.array(y)
