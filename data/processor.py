# stock_data_processor.py
import numpy as np

class StockDataProcessor:
    def __init__(self, window_size=30, pred_days=3):
        self.window_size = window_size
        self.pred_days = pred_days

    def create_sliding_window(self, data):
        """
        Generate sliding windows for training.
        The data will include multiple features (open, high, low, close, volume).
        """
        X, y = [], []
        for i in range(len(data) - self.window_size - self.pred_days):
            X.append(data[i: i + self.window_size, :])  # Include all features in the window
            y.append(data[i + self.window_size: i + self.window_size + self.pred_days, 3])  # Use the 'Close' price for prediction
        return np.array(X), np.array(y)

    def preprocess(self, stock_data):
        """
        Preprocess the stock data by generating sliding windows of time-series data.
        Include multiple features (OHLCV).
        """
        values = stock_data.values  # Use all available time-series data
        return self.create_sliding_window(values)
