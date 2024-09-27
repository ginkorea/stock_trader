from models.regression.regression_model import RegressionModel
from data.processor.processors import RegressionProcessor as StockDataRegressionProcessor
from pipelines.base_pipeline import BasePipeline


class RegressionPipeline(BasePipeline):
    def __init__(self, start_date, end_date, window_size, pred_days, ticker=None):
        """
        Initializes the Regression Pipeline.
        """
        super().__init__(start_date, end_date, window_size, pred_days, ticker)
        self.regression_model = None

    def train_model(self):
        """
        Fetches and preprocesses data, and trains the regression model.
        """
        # Fetch and preprocess the data
        preprocessed_data = self.fetch_and_preprocess_data(StockDataRegressionProcessor)

        # Initialize the regression model
        self.regression_model = RegressionModel()

        # Fit the model using preprocessed data
        regression_results = self.regression_model.fit(preprocessed_data)

        print("Regression model training completed.")
        return regression_results

    def predict(self, ticker, window, X_new):
        """
        Makes predictions using the trained regression model.

        Args:
            ticker (str): The stock ticker to predict.
            window (int): The time window to use (5, 15, 30, 90).
            X_new (array-like): New open price data to make predictions.

        Returns:
            Predicted values for the given ticker and window.
        """
        # Ensure that the model has been trained
        if not self.regression_model:
            raise ValueError("Model has not been trained yet. Please call `train_model` before making predictions.")

        # Use the trained model to make predictions
        predicted_high = self.regression_model.predict(ticker, window, X_new)

        print(f"Predicted high prices for {ticker} over {window}-day window: {predicted_high}")
        return predicted_high
