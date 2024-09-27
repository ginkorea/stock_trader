from models.transformer.pipeline import Pipeline
import pandas as pd


def predict_single_ticker(ticker, start_date, end_date, window_size, pred_days):
    """
    Predicts stock prices for a single ticker and returns the results.

    Args:
        ticker (str): The stock ticker to predict.
        start_date (str): Start date for stock data.
        end_date (str): End date for stock data.
        window_size (int): Number of days to consider for training.
        pred_days (int): Days ahead to predict.

    Returns:
        list: List of predicted values for the given ticker.
    """
    # Initialize the pipeline with the single ticker
    pipeline = Pipeline(start_date=start_date, end_date=end_date, window_size=window_size, pred_days=pred_days,
                        ticker=ticker, model_save_path=f"models/{ticker}_model.pth")

    # Run the prediction and return the results
    try:
        predictions = pipeline.predict()
        return predictions
    except ValueError as e:
        print(f"Error during prediction for {ticker}: {e}")
        return []




def batch_predict_tickers(_tickers, start_date, end_date, window_size, pred_days):
    """
    Predicts stock prices for a list of tickers and ranks them by day.

    Args:
        _tickers (list): List of stock tickers to predict.
        start_date (str): Start date for stock data.
        end_date (str): End date for stock data.
        window_size (int): Number of days to consider for training.
        pred_days (int): Days ahead to predict.

    Returns:
        pd.DataFrame: DataFrame containing the rank and predicted values of stocks by day.
    """
    results_by_day = {}

    for ticker in _tickers:
        print(f"Starting prediction for {ticker}")
        try:
            predictions = predict_single_ticker(ticker, start_date, end_date, window_size, pred_days)

            # Store predictions by day for ranking
            try:
                for day, value in enumerate(predictions):
                    if day not in results_by_day:
                        results_by_day[day] = []
                    results_by_day[day].append((ticker, value))
            except TypeError as e:
                print(f"Error extracting predictions for {ticker}: {e}")
                print("Continuing with next ticker...")
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

    # Predict and rank stock prices for all tickers
    ranked_df = batch_predict_tickers(_tickers=tickers, start_date="2022-10-01", end_date="2022-10-15",
                                      window_size=5, pred_days=1)

    # Save the ranked predictions to a CSV file for further analysis
    ranked_df.to_csv("ranked_predictions_by_day.csv", index=False)

    print(ranked_df)
