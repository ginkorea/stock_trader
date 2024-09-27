from pipelines.pipelines import TransformerPipeline as Pipeline



def train_single_ticker(ticker, start_date, end_date, window_size, pred_days, learning_rate, batch_size, epochs):
    """
    Trains a model for a single ticker and saves the model with the ticker name.

    Args:
        ticker (str): The stock ticker to train the model on.
        start_date (str): Start date for stock data.
        end_date (str): End date for stock data.
        window_size (int): Number of days to consider for training.
        pred_days (int): Days ahead to predict.
        learning_rate (float): Learning rate for the training process.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
    """
    # Initialize the pipeline with the single ticker
    pipeline = Pipeline(start_date=start_date, end_date=end_date, window_size=window_size, pred_days=pred_days,
                        ticker=ticker, model_save_path=f"weights/{ticker}_model.pth")

    # Train the model for this ticker
    pipeline.train_model(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
    print(f"Training completed for {ticker}")

def batch_train_tickers(_tickers, start_date, end_date, window_size, pred_days, learning_rate, batch_size, epochs):
    """
    Trains weights for a list of tickers and saves each model with the ticker name.

    Args:
        _tickers (list): List of stock tickers to train weights on.
        start_date (str): Start date for stock data.
        end_date (str): End date for stock data.
        window_size (int): Number of days to consider for training.
        pred_days (int): Days ahead to predict.
        learning_rate (float): Learning rate for the training process.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
    """
    for ticker in _tickers:
        print(f"Starting training for {ticker}")
        try:
            train_single_ticker(ticker, start_date, end_date, window_size, pred_days, learning_rate, batch_size, epochs)
        except ValueError as e:
            print(f"Error during training for {ticker}: {e}")
            print("Continuing with next ticker...")
    print("Batch training completed.")


if __name__ == "__main__":

    from data.tickers import tickers

    # Train weights for all tickers
    batch_train_tickers(_tickers=tickers, start_date="2022-01-01", end_date="2022-10-01",
                        window_size=5, pred_days=1, learning_rate=0.0001,
                        batch_size=32, epochs=500)