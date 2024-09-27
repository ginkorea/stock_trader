from transformer.model import TransformerModel
from data.processor import StockDataProcessor
from data.fetcher import StockDataFetcher
from data.tickers import tickers as tick
from connection.client import APIConnection
from transformer.trainer import Trainer
from transformer.inference import Inference
from utils.model_size import verify_labels

class Pipeline:
    def __init__(self, start_date, end_date, window_size, pred_days, tickers=None, model_save_path="stock_model.pth"):
        self.api_connection = APIConnection()
        self.tickers = tick if tickers is None else tickers
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = window_size
        self.pred_days = pred_days
        self.model_save_path = model_save_path

    def fetch_and_preprocess_data(self):

        # Fetch stock data
        data_fetcher = StockDataFetcher(self.api_connection, start_date=self.start_date, end_date=self.end_date)
        stock_data = data_fetcher.fetch_data(self.tickers)

        # Check if data is empty or incomplete
        if stock_data.empty:
            raise ValueError("Fetched data is empty. Please check the stock symbols or date range.")

        # Preprocess the data to create long vectors per day, pad, and calculate input size
        data_processor = StockDataProcessor(window_size=self.window_size, pred_days=self.pred_days)
        X, y, tickers, input_size, num_heads, hidden_dim = data_processor.preprocess(stock_data, self.tickers)

        if X.size == 0:
            raise ValueError("Preprocessed data is empty. Check if there are valid trading days in the given range.")

        print("Data preprocessing complete.")
        print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
        print(f"Input size: {input_size}, Number of heads: {num_heads}, Hidden dimension: {hidden_dim}")
        return X, y, tickers, input_size, num_heads, hidden_dim

    @staticmethod
    def setup_model(input_size, num_heads, hidden_dim, num_tickers):
        # Initialize the Transformer model
        model = TransformerModel(input_size=input_size, num_heads=num_heads, num_layers=4, hidden_dim=hidden_dim, num_tickers=num_tickers)
        return model

    def train_model(self, learning_rate=0.001, batch_size=32, epochs=50):
        X, y, tickers, input_size, num_heads, hidden_dim = self.fetch_and_preprocess_data()

        # Verify the labels
        verify_labels(y)

        # Initialize the model
        model = self.setup_model(input_size, num_heads, hidden_dim, len(tickers))

        # Initialize the trainer
        trainer = Trainer(model=model, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs, model_save_path=self.model_save_path)

        # Train the model
        trainer.train(X, y)

    def predict(self):
        X, _, tickers, input_size, num_heads, hidden_dim = self.fetch_and_preprocess_data()

        # Initialize the model
        model = self.setup_model(input_size, num_heads, hidden_dim, len(tickers))

        # Initialize the inference engine
        inference_engine = Inference(model=model, model_path=self.model_save_path)

        # Perform inference and get the top 5 tickers
        top_tickers, top_scores = inference_engine.predict(X, tickers)

        # Display the top 5 tickers and their predicted ratios
        for i in range(len(top_tickers)):
            print(f"Prediction for day {i + 1}:")
            for ticker, score in zip(top_tickers[i], top_scores[i]):
                print(f"Ticker: {ticker}, Predicted Ratio: {score:.4f}")
