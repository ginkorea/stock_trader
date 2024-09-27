import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    def __init__(self, model, learning_rate, batch_size, epochs, model_save_path, weight_decay=1e-5):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_decay = weight_decay

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  # Added weight decay
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        patience = 5  # Early stopping patience
        counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(data_loader)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}")

            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break