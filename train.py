from transformer.trainer import Trainer
from transformer.model import TransformerModel
from utils.model_size import find_best_head_size, pad_input, verify_labels
from data.processor import StockDataProcessor
from data.fetcher import StockDataFetcher
from data.tickers import tickers as tick  # Import the list of tickers
from connection.client import APIConnection


# Initialize the API connection (assuming it's available as an import or in the environment)
api_connection = APIConnection()
tickers = tick  # List of tickers to fetch data for

# Fetch stock data for all tickers
data_fetcher = StockDataFetcher(api_connection, start_date="2022-01-01", end_date="2023-01-01")
stock_data = data_fetcher.fetch_data(tickers)

# Preprocess the data to create long vectors per day, pad, and calculate input size
data_processor = StockDataProcessor(window_size=30, pred_days=1)  # Only predicting the next day high
X, y, tickers, input_size, num_heads, hidden_dim = data_processor.preprocess(stock_data, tickers)

# Initialize the Transformer model
model = TransformerModel(input_size=input_size, num_heads=num_heads, num_layers=6, hidden_dim=hidden_dim, num_tickers=len(tickers))

# Initialize the trainer
trainer = Trainer(model=model, learning_rate=0.001, batch_size=32, epochs=50, model_save_path="stock_model.pth")

# Verify the labels
verify_labels(y)

# Train the model
trainer.train(X, y)
