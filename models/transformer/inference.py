import torch

class Inference:
    def __init__(self, model, model_path, device=None, verbose=False):
        self.model = model
        self.model_path = model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.model.eval()
        self.load_model()
        self.verbose = verbose

    def load_model(self):
        """ Loads the saved model weights. """
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def predict(self, X):
        """ Predicts a continuous value for the input data. """
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.model(X)

            # Debugging: Print the raw model output to inspect the values
            if self.verbose:
                print(f"Model output: {output.cpu().numpy()}")

            return output.cpu().numpy()  # Return the predicted values
