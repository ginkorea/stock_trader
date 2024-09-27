import torch


class Inference:
    def __init__(self, model, model_path, device=None):
        self.model = model
        self.model_path = model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.model.eval()
        self.load_model()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def predict(self, X, tickers):
        with torch.no_grad():
            # Debugging: Print the shape and first input batch to check input variability
            print(f"Input shape: {X.shape}")
            print(f"First input batch: {X[0]}")
            print(f"Last input batch: {X[-1]}")

            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.model(X)

            # Debugging: Print raw model output to inspect values before applying softmax
            print(f"Raw model output (first batch): {output[0].cpu().numpy()}")
            print(f"Raw model output (last batch): {output[-1].cpu().numpy()}")

            output = torch.softmax(output, dim=1)  # Use softmax for probabilistic ranking

            # Debugging: Print output after softmax
            print(f"Softmax model output (first batch): {output[0].cpu().numpy()}")
            print(f"Softmax model output (last batch): {output[-1].cpu().numpy()}")

            # Get the top 5 predicted tickers and their scores
            sorted_indices = torch.argsort(output, dim=1, descending=True)
            top_5_indices = sorted_indices[:, :5]  # Get top 5 tickers
            top_5_scores = torch.gather(output, 1, top_5_indices)

            # Debugging: Print top 5 tickers and their scores
            print(f"Top 5 indices (first batch): {top_5_indices[0].cpu().numpy()}")
            print(f"Top 5 scores (first batch): {top_5_scores[0].cpu().numpy()}")

            top_5_tickers = [[tickers[idx] for idx in indices] for indices in top_5_indices.tolist()]
            top_5_scores = top_5_scores.tolist()

            return top_5_tickers, top_5_scores
