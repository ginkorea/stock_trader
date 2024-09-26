# main.py
from connection.client import APIConnection
from data.fetcher import StockDataFetcher
from data.processor import StockDataProcessor
from transformer.model import TransformerModel
from transformer.trainer import Trainer

# Step 1: Initialize Alpaca API Connection
api_connection = APIConnection(is_paper=True)

# Step 2: Fetch stock data using StockDataFetcher
data_fetcher = StockDataFetcher(api_connection, start_date="2020-01-01", end_date="2023-01-01")
ticker_data = data_fetcher.fetch_data('AAPL')  # Fetch data for a single stock (OHLCV)

# Step 3: Preprocess the data (with all features)
data_processor = StockDataProcessor(window_size=30, pred_days=3)
X, y = data_processor.preprocess(ticker_data)  # X will now include OHLCV features

# Step 4: Initialize the Transformer model (with input_size=5 for OHLCV)
model = TransformerModel(input_size=5, num_heads=2, num_layers=3, hidden_dim=64, output_dim=3)

# Step 5: Train the model
trainer = Trainer(model=model, learning_rate=0.001, batch_size=32, epochs=50)
trainer.train(X, y)

# Step 6: Make predictions
test_data = data_fetcher.fetch_data('AAPL')  # Fetch new data for predictions
X_test, _ = data_processor.preprocess(test_data)
predictions = trainer.predict(X_test)
print(predictions)
