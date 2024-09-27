from models.transformer.model import TransformerModel
from models.transformer.trainer import Trainer
from models.transformer.inference import Inference
from utils.model_size import verify_labels
from data.processor.processors import TransformerProcessor as StockDataTransformerProcessor
from pipelines.base_pipeline import BasePipeline


class TransformerPipeline(BasePipeline):
    def __init__(self, start_date, end_date, window_size, pred_days, ticker=None, model_save_path="stock_model.pth"):
        """
        Initializes the Transformer Pipeline.
        """
        super().__init__(start_date, end_date, window_size, pred_days, ticker)
        self.model_save_path = model_save_path

    def train_model(self, learning_rate=0.001, batch_size=32, epochs=50):
        """
        Trains the Transformer model on the preprocessed data.

        Args:
            learning_rate (float): Learning rate for training.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs to train.
        """
        X, y, tickers, input_size, num_heads, hidden_dim = self.fetch_and_preprocess_data(StockDataTransformerProcessor)

        # Verify the labels
        verify_labels(y)

        # Initialize the model
        model = TransformerModel(input_size=input_size, num_heads=num_heads, num_layers=4, hidden_dim=hidden_dim)

        # Initialize the trainer
        trainer = Trainer(model=model, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs,
                          model_save_path=self.model_save_path)

        # Train the model
        trainer.train(X, y)

    def predict(self):
        """
        Runs the prediction using the trained Transformer model.
        """
        X, _, tickers, input_size, num_heads, hidden_dim = self.fetch_and_preprocess_data(StockDataTransformerProcessor)

        # Initialize the model
        model = TransformerModel(input_size=input_size, num_heads=num_heads, num_layers=4, hidden_dim=hidden_dim)

        # Initialize the inference engine
        inference_engine = Inference(model=model, model_path=self.model_save_path)

        # Perform inference and get the predictions
        predictions = inference_engine.predict(X)

        # Display predictions
        for i, pred in enumerate(predictions):
            print(f"Prediction for day {i + 1}: Predicted Value: {pred}")

        return predictions
