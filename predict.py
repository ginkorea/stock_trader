from pipelines.pipelines import TransformerPipeline, RegressionPipeline
import pandas as pd

def predict_single_ticker(ticker, start_date, end_date, window_size, pred_days, _model_type):
    """
    Predicts stock prices for a single ticker and returns the results based on the model type (Transformer or Regression).

    Args:
        ticker (str): The stock ticker to predict.
        start_date (str): Start date for stock data.
        end_date (str): End date for stock data.
        window_size (int): Number of days to consider for training.
        pred_days (int): Days ahead to predict.
        _model_type (str): The type of model to use ('transformer' or 'regression').

    Returns:
        list: List of predicted values for the given ticker.
    """
    if _model_type == 'transformer':
        pipeline = TransformerPipeline(start_date=start_date, end_date=end_date, window_size=window_size, pred_days=pred_days,
                                       ticker=ticker, model_save_path=f"weights/transformer/{ticker}_model.pth")
    elif _model_type == 'regression':
        pipeline = RegressionPipeline(start_date=start_date, end_date=end_date, window_size=window_size, pred_days=pred_days,
                                      ticker=ticker)
    else:
        raise ValueError("Invalid model_type. Please choose 'transformer' or 'regression'.")

    # Run the prediction and return the results
    try:
        predictions = pipeline.predict()  # Replace with actual new data
        return predictions
    except ValueError as e:
        print(f"Error during prediction for {ticker}: {e}")
        return []


def batch_predict_tickers(_tickers, start_date, end_date, window_size, pred_days, model_type):
    """
    Predicts stock prices for a list of tickers and ranks them by day based on the model type (Transformer or Regression).

    Args:
        _tickers (list): List of stock tickers to predict.
        start_date (str): Start date for stock data.
        end_date (str): End date for stock data.
        window_size (int): Number of days to consider for training.
        pred_days (int): Days ahead to predict.
        model_type (str): The type of model to use ('transformer' or 'regression').

    Returns:
        pd.DataFrame: DataFrame containing the rank and predicted values of stocks by day.
    """
    results_by_day = {}

    for ticker in _tickers:
        print(f"Starting prediction for {ticker} using {model_type} model")
        try:
            predictions = predict_single_ticker(ticker, start_date, end_date, window_size, pred_days, model_type)

            # Store predictions by day for ranking
            for day, value in enumerate(predictions):
                if day not in results_by_day:
                    results_by_day[day] = []
                results_by_day[day].append((ticker, value))
        except ValueError as e:
            print(f"Error during prediction for {ticker}: {e}")
            print("Continuing with next ticker...")

    # Rank the tickers by their predicted values for each day
    ranked_results = []
    for day, ticker_values in results_by_day.items():
        ranked_ticker_values = sorted(ticker_values, key=lambda x: x[1], reverse=True)  # Sort by value
        for rank, (ticker, value) in enumerate(ranked_ticker_values, 1):
            ranked_results.append({"Day": day + 1, "Ticker": ticker, "Predicted Value": value, "Rank": rank})

    # Convert the ranked results into a DataFrame
    return pd.DataFrame(ranked_results)


if __name__ == "__main__":
    from data.tickers import tickers

    # Choose model type ('transformer' or 'regression')
    model_type = 'transformer'  # Change to 'regression' to predict using the regression model

    # Predict and rank stock prices for all tickers
    ranked_df = batch_predict_tickers(_tickers=tickers, start_date="2022-10-01", end_date="2022-10-15",
                                      window_size=5, pred_days=1, model_type=model_type)

    # Save the ranked predictions to a CSV file for further analysis
    ranked_df.to_csv("ranked_predictions_by_day.csv", index=False)

    print(ranked_df)
