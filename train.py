from pipelines.pipelines import RegressionPipeline, TransformerPipeline


def train_single_ticker(ticker, start_date, end_date, window_size, pred_days, learning_rate, batch_size, epochs,
                        model_type):
    """
    Trains a model for a single ticker based on the selected model type (Transformer or Regression).

    Args:
        ticker (str): The stock ticker to train the model on.
        start_date (str): Start date for stock data.
        end_date (str): End date for stock data.
        window_size (int): Number of days to consider for training.
        pred_days (int): Days ahead to predict.
        learning_rate (float): Learning rate for the training process (only for Transformer).
        batch_size (int): Batch size for training (only for Transformer).
        epochs (int): Number of epochs for training (only for Transformer).
        model_type (str): The type of model to use ('transformer' or 'regression').
    """
    if model_type == 'transformer':
        # Initialize the Transformer pipeline
        pipeline = TransformerPipeline(start_date=start_date, end_date=end_date, window_size=window_size,
                                       pred_days=pred_days, ticker=ticker)
        # Train the Transformer model
        pipeline.train_model(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
    elif model_type == 'regression':
        # Initialize the Regression pipeline
        pipeline = RegressionPipeline(start_date=start_date, end_date=end_date, window_size=window_size,
                                      pred_days=pred_days, ticker=ticker)
        # Train the Regression model
        pipeline.train_model()
    else:
        raise ValueError("Invalid model_type. Please choose 'transformer' or 'regression'.")

    print(f"Training completed for {ticker} using {model_type} model.")


def batch_train_tickers(_tickers, start_date, end_date, window_size, pred_days, learning_rate, batch_size, epochs,
                        model_type):
    """
    Trains models for a list of tickers and saves each model based on the selected model type (Transformer or Regression).

    Args:
        _tickers (list): List of stock tickers to train models for.
        start_date (str): Start date for stock data.
        end_date (str): End date for stock data.
        window_size (int): Number of days to consider for training.
        pred_days (int): Days ahead to predict.
        learning_rate (float): Learning rate for the training process (only for Transformer).
        batch_size (int): Batch size for training (only for Transformer).
        epochs (int): Number of epochs for training (only for Transformer).
        model_type (str): The type of model to use ('transformer' or 'regression').
    """
    for ticker in _tickers:
        print(f"Starting training for {ticker} with {model_type} model")
        try:
            train_single_ticker(ticker, start_date, end_date, window_size, pred_days, learning_rate, batch_size, epochs,
                                model_type)
        except ValueError as e:
            print(f"Error during training for {ticker}: {e}")
            print("Continuing with next ticker...")

    print("Batch training completed.")


if __name__ == "__main__":
    from data.tickers import tickers

    # Choose model type ('transformer' or 'regression')
    model_type = 'transformer'  # Change to 'regression' to train using the regression model

    # Train models for all tickers
    batch_train_tickers(_tickers=tickers, start_date="2022-01-01", end_date="2022-10-01",
                        window_size=5, pred_days=1, learning_rate=0.0001,
                        batch_size=32, epochs=500, model_type=model_type)
