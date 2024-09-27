from models.transformer.model import TransformerModel
from data.processor.processors import TransformerProcessor
from data.fetcher import StockDataFetcher
from connection.client import APIConnection
from models.transformer.trainer import Trainer
from models.transformer.inference import Inference
from utils.model_size import verify_labels


class Pipeline:
    def __init__(self, start_date, end_date, window_size, pred_days, ticker=None, model_save_path="stock_model.pth"):
        """
        Initializes the Pipeline for a single ticker or multiple tickers.

        Args:
            start_date (str): Start date for stock data.
            end_date (str): End date for stock data.
            window_size (int): Number of days to consider for training.
            pred_days (int): Days ahead to predict.
            ticker (str or list): Ticker symbol or list of ticker symbols.
            model_save_path (str): Path to save the trained model.
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
        self.model_save_path = model_save_path

    def fetch_and_preprocess_data(self):
        """
        Fetches and preprocesses stock data for the provided tickers.
        """
        # Fetch stock data
        data_fetcher = StockDataFetcher(self.api_connection, start_date=self.start_date, end_date=self.end_date)
        stock_data = data_fetcher.fetch_data(self.tickers)

        # Check if data is empty or incomplete
        if stock_data.empty:
            raise ValueError(
                f"Fetched data for tickers {self.tickers} is empty. Please check the stock symbol(s) or date range.")

        # Preprocess the data to create long vectors per day, pad, and calculate input size
        data_processor = TransformerProcessor(window_size=self.window_size, pred_days=self.pred_days)
        X, y, tickers, input_size, num_heads, hidden_dim = data_processor.preprocess(stock_data, self.tickers)

        if X.size == 0:
            raise ValueError("Preprocessed data is empty. Check if there are valid trading days in the given range.")

        print("Data preprocessing complete.")
        print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
        print(f"Input size: {input_size}, Number of heads: {num_heads}, Hidden dimension: {hidden_dim}")
        return X, y, tickers, input_size, num_heads, hidden_dim

    @staticmethod
    def setup_model(input_size, num_heads, hidden_dim, num_tickers):
        """
        Initializes the Transformer model.

        Args:
            input_size (int): Size of the input data.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension size.
            num_tickers (int): Number of tickers to predict for.
        """
        model = TransformerModel(input_size=input_size, num_heads=num_heads, num_layers=4, hidden_dim=hidden_dim)
        return model

    def train_model(self, learning_rate=0.001, batch_size=32, epochs=50):
        """
        Trains the Transformer model on the preprocessed data.

        Args:
            learning_rate (float): Learning rate for training.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs to train.
        """
        X, y, tickers, input_size, num_heads, hidden_dim = self.fetch_and_preprocess_data()

        # Verify the labels
        verify_labels(y)

        # Initialize the model
        model = self.setup_model(input_size, num_heads, hidden_dim, len(tickers))

        # Initialize the trainer
        trainer = Trainer(model=model, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs,
                          model_save_path=self.model_save_path)

        # Train the model
        trainer.train(X, y)

    def predict(self):
        """
        Runs the prediction using the trained model for a single ticker.
        """
        # Fetch and preprocess the data
        X, _, tickers, input_size, num_heads, hidden_dim = self.fetch_and_preprocess_data()

        # Initialize the model
        model = self.setup_model(input_size, num_heads, hidden_dim, len(tickers))

        # Initialize the inference engine
        inference_engine = Inference(model=model, model_path=self.model_save_path)

        # Perform inference and get the predictions
        predictions = inference_engine.predict(X)

        # Display predictions
        for i, pred in enumerate(predictions):
            print(f"Prediction for day {i + 1}: Predicted Value: {pred}")

        return predictions

