from abc import ABC, abstractmethod

class StockDataProcessor(ABC):
    def __init__(self, window_size, pred_days, verbose=False):
        """
        Base processor with window and prediction size.

        Args:
            window_size (int): Number of days to use in the window for prediction.
            pred_days (int): Number of days ahead to predict.
        """
        self.window_size = window_size
        self.pred_days = pred_days
        self.verbose = verbose

    def extract_features(self, stock_data, tickers):
        """
        Extracts features for each day and ticker, and handles missing data.

        Args:
            stock_data (pd.DataFrame): DataFrame containing the stock data.
            tickers (list): List of ticker symbols.

        Returns:
            list: List of feature vectors for each day.
        """
        all_features = []

        if self.verbose:
            print(stock_data.head(), stock_data.columns)  # For debugging purposes

        # Sort by date for proper sequencing
        stock_data = stock_data.sort_values('timestamp')

        # Group by 'timestamp' to collect data by date
        grouped_data = stock_data.groupby('timestamp')

        # Process the data day by day
        for timestamp, group in grouped_data:
            day_vector = []

            # Loop through each ticker to extract features
            for ticker in tickers:
                company_data = group[group['symbol'] == ticker]

                if not company_data.empty:
                    # Extract 7 relevant features (e.g., open, high, etc.)
                    day_vector.extend(company_data.iloc[0, 2:].values.tolist())
                else:
                    # Fill missing data with zeros
                    day_vector.extend([0] * 7)

            # Ensure that the day vector is complete
            if len(day_vector) == 7 * len(tickers):
                all_features.append(day_vector)

        return all_features

    @abstractmethod
    def preprocess(self, stock_data, tickers):
        """
        Abstract method to be implemented by subclasses for data preprocessing.

        Args:
            stock_data (pd.DataFrame): DataFrame containing the stock data.
            tickers (list): List of ticker symbols.
        """
        pass

    @staticmethod
    def check_valid_sequences(X):
        """
        Checks if valid sequences are found in the preprocessed data.

        Args:
            X (np.array): Input data sequences.

        Raises:
            ValueError: If no valid sequences are found.
        """
        if X.shape[0] == 0:
            raise ValueError("No valid sequences found during preprocessing.")
