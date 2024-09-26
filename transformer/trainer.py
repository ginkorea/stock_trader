# trainer.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self, model, learning_rate=0.001, batch_size=32, epochs=50):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, X_train, y_train):
        """Train the model on the given data."""
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(train_loader)}')

    def predict(self, x_test):
        """Make predictions using the trained model."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(x_test, dtype=torch.float32))
        return predictions
