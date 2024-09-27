import numpy as np
from sklearn.linear_model import LinearRegression

class RegressionModel:
    def __init__(self):
        """
        Initializes the RegressionModel class.
        """
        self.models = {}

    def fit(self, preprocessed_data):
        """
        Fits a linear regression model for each ticker over multiple time windows.

        Args:
            preprocessed_data (dict): Preprocessed open-high data.

        Returns:
            dict: Dictionary of regression results for each ticker and time window.
        """
        regression_results = {}

        # Perform regression for each ticker and time window
        for ticker, window_data in preprocessed_data.items():
            self.models[ticker] = {}
            regression_results[ticker] = {}

            for window, data in window_data.items():
                X, y = data["X"], data["y"]
                model = LinearRegression()

                # Fit the regression model
                model.fit(X, y)

                # Store the coefficients and intercept
                self.models[ticker][window] = model
                regression_results[ticker][window] = {
                    "coefficients": model.coef_,
                    "intercept": model.intercept_
                }

                print(f"Regression for {ticker}, {window}-day window completed.")
                print(f"Coefficients: {model.coef_}, Intercept: {model.intercept_}")

        return regression_results

    def predict(self, ticker, window, X_new):
        """
        Makes a prediction for the given ticker and window using new data.

        Args:
            ticker (str): The stock ticker to predict.
            window (int): The time window to use (5, 15, 30, 90).
            X_new (array-like): New open price data to make predictions.

        Returns:
            array: Predicted high prices.
        """
        if ticker not in self.models or window not in self.models[ticker]:
            raise ValueError(f"No regression model found for {ticker} with a {window}-day window.")

        model = self.models[ticker][window]

        # Perform prediction using the regression model
        X_new = np.array(X_new).reshape(1, -1)  # Ensure X_new is 2D
        y_pred = model.predict(X_new)

        return y_pred
