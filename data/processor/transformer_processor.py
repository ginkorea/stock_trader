from data.processor.base import StockDataProcessor
from utils.model_size import find_best_head_size, pad_input
import numpy as np

class StockDataTransformerProcessor(StockDataProcessor):
    def preprocess(self, stock_data, tickers):
        """
        Preprocesses data specifically for the Transformer model.

        Args:
            stock_data (pd.DataFrame): DataFrame containing the stock data.
            tickers (list): List of ticker symbols.

        Returns:
            Processed data ready for Transformer training.
        """
        # Extract features for each day and ticker
        all_features = self.extract_features(stock_data, tickers)

        # Create input-output sequences based on window size
        X, y = self.create_sequences(all_features, tickers)

        # Check for valid sequences
        self.check_valid_sequences(X)

        # Adjust the data for Transformer model requirements
        X_padded, input_size, num_heads, hidden_dim = self.adjust_for_transformer(X)

        return X_padded, y, tickers, input_size, num_heads, hidden_dim

    def create_sequences(self, all_features, tickers):
        """
        Creates input-output sequences from the feature data.

        Args:
            all_features (list): List of feature vectors for each day.
            tickers (list): List of ticker symbols.

        Returns:
            X (list): Input sequences for the model.
            y (list): Target output labels for the model.
        """
        X = []
        y = []

        # Create input-output sequences based on window size
        for i in range(len(all_features) - self.window_size - self.pred_days):
            X.append(all_features[i:i + self.window_size])  # Collect a window of days

            # Calculate prediction labels based on high/open price ratio
            day_pred = self.calculate_day_prediction(all_features, i, tickers)
            y.append(day_pred)

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        return X, y

    def calculate_day_prediction(self, all_features, index, tickers):
        """
        Calculates the prediction label based on the ratio of high/open prices.

        Args:
            all_features (list): List of feature vectors for each day.
            index (int): Current index in the feature vector list.
            tickers (list): List of ticker symbols.

        Returns:
            list: Normalized ratio of high/open price for each ticker.
        """
        day_pred = []
        for j in range(len(tickers)):
            high = all_features[index + self.window_size + self.pred_days][j * 7 + 1]  # High price
            open_ = all_features[index + self.window_size + self.pred_days][j * 7]  # Open price
            ratio = high / open_ if open_ != 0 else 0  # Avoid division by zero
            day_pred.append(ratio)

        # Normalize ratios between 0 and 1
        min_ratio = min(day_pred)
        max_ratio = max(day_pred)
        if max_ratio - min_ratio != 0:
            normalized_pred = [(r - min_ratio) / (max_ratio - min_ratio) for r in day_pred]
        else:
            normalized_pred = day_pred  # Handle case where all values are the same

        return normalized_pred

    @staticmethod
    def adjust_for_transformer(X):
        """
        Adjusts the input data for use in a Transformer model, calculating the number of attention heads
        and padding the input data if necessary.

        Args:
            X (np.array): Input data sequences.

        Returns:
            X_padded (np.array): Padded input data.
            input_size (int): Size of the input data.
            num_heads (int): Best number of heads for multi-head attention.
            hidden_dim (int): Hidden dimension size for the Transformer.
        """
        input_size = X.shape[-1]
        num_heads = find_best_head_size(input_size)
        hidden_dim = input_size * 2
        if hidden_dim % num_heads != 0:
            padding_size = num_heads - (hidden_dim % num_heads)
            X_padded, new_input_size = pad_input(X, input_size + padding_size)
            hidden_dim = new_input_size * 2
        else:
            X_padded, new_input_size = X, input_size
        return X_padded, new_input_size, num_heads, hidden_dim
