import numpy as np
from utils.model_size import find_best_head_size, pad_input

class StockDataProcessor:
    def __init__(self, window_size, pred_days):
        self.window_size = window_size
        self.pred_days = pred_days

    def preprocess(self, stock_data, tickers):
        # Assuming 'stock_data' contains rows with 'symbol', 'timestamp', 'open', 'high', and other features
        all_features = []
        print(stock_data.head(), stock_data.columns)

        # Sort by date for proper sequencing
        stock_data = stock_data.sort_values('timestamp')

        # Group by 'timestamp' to collect all companies' data on the same day
        grouped_data = stock_data.groupby('timestamp')

        for timestamp, group in grouped_data:
            day_vector = []

            # Append features for each company in the same order as the tickers list
            for ticker in tickers:
                company_data = group[group['symbol'] == ticker]
                if not company_data.empty:
                    day_vector.extend(company_data.iloc[0, 2:].values.tolist())  # Get the 7 features per ticker
                else:
                    day_vector.extend([0] * 7)  # Append zeros if data for the ticker is missing

            if len(day_vector) == 7 * len(tickers):  # Ensure the vector length is correct
                all_features.append(day_vector)

        X = []
        y = []

        # Convert the day vectors into sequences with 'window_size'
        for i in range(len(all_features) - self.window_size - self.pred_days):
            X.append(all_features[i:i + self.window_size])  # Window of 30 days

            # Predict based on ratio of high/open, normalize between 0 and 1
            day_pred = []
            for j in range(len(tickers)):
                high = all_features[i + self.window_size + self.pred_days][j * 7 + 1]  # High price
                open_ = all_features[i + self.window_size + self.pred_days][j * 7]  # Open price
                ratio = high / open_ if open_ != 0 else 0
                day_pred.append(ratio)

            # Normalize ratios between 0 and 1
            min_ratio = min(day_pred)
            max_ratio = max(day_pred)
            if max_ratio - min_ratio != 0:
                normalized_pred = [(r - min_ratio) / (max_ratio - min_ratio) for r in day_pred]
            else:
                normalized_pred = day_pred  # Avoid division by zero if all ratios are the same

            y.append(normalized_pred)

        X = np.array(X)
        y = np.array(y)

        # Check for valid sequences
        if X.shape[0] == 0:
            raise ValueError("No valid sequences found during preprocessing.")

        # Determine input size before padding
        input_size = X.shape[-1]

        # Calculate the best number of heads for multi-head attention
        num_heads = find_best_head_size(input_size)

        # Ensure hidden_dim is divisible by num_heads, adjust if necessary
        hidden_dim = input_size * 2
        if hidden_dim % num_heads != 0:
            padding_size = num_heads - (hidden_dim % num_heads)
            X_padded, new_input_size = pad_input(X, input_size + padding_size)
            hidden_dim = new_input_size * 2  # Update hidden_dim after padding
        else:
            X_padded, new_input_size = X, input_size  # No padding needed

        return X_padded, y, tickers, new_input_size, num_heads, hidden_dim
