from data.processor.processors import RegressionProcessor
from pipelines.base_pipeline import BasePipeline


class RegressionPipeline(BasePipeline):
    def __init__(self, start_date, end_date, window_size, pred_days, ticker=None):
        """
        Initializes the Regression Pipeline.
        """
        super().__init__(start_date, end_date, window_size, pred_days, ticker)

    def train_model(self):
        """
        Preprocess data for regression and return the open-high sequences.
        """
        open_high_data = self.fetch_and_preprocess_data(RegressionProcessor)
        print("Regression data extracted successfully.")
        return open_high_data

    def predict(self):
        """
        Prediction is handled externally in the regression pipeline.
        """
        print("This pipeline doesn't perform predictions. Use the extracted data for regression models.")
        return None
