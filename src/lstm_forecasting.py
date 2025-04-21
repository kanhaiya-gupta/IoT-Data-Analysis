import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_length: int):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.data) - self.seq_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.seq_length].view(-1, 1)  # Shape: (seq_length, 1)
        y = self.data[idx + self.seq_length]  # Shape: (1,)
        return x, y

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # out shape: (batch_size, output_size)
        return out

class LSTMForecasting:
    def __init__(self, 
                 seq_length: int = 24,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 learning_rate: float = 0.001,
                 num_epochs: int = 100,
                 batch_size: int = 32):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def prepare_data(self, data: pd.DataFrame, target_column: str) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for LSTM training"""
        # Scale the data while preserving feature names
        data_to_scale = data[[target_column]].copy()
        scaled_data = pd.DataFrame(
            self.scaler.fit_transform(data_to_scale),
            columns=data_to_scale.columns
        ).values.flatten()
        
        # Create sequences
        dataset = TimeSeriesDataset(scaled_data, self.seq_length)
        
        # Split into train and test
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

    def train(self, train_loader: DataLoader) -> List[float]:
        """Train the LSTM model"""
        input_size = 1  # Number of features
        output_size = 1  # Number of predictions
        
        self.model = LSTMForecaster(input_size, self.hidden_size, self.num_layers, output_size)
        self.model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                # Reshape input: (batch_size, seq_length, input_size)
                batch_X = batch_X.view(batch_X.size(0), -1, 1).to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
        
        return losses

    def predict(self, sequence: np.ndarray) -> float:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.model.eval()
        with torch.no_grad():
            # Scale the input sequence with feature names
            sequence_df = pd.DataFrame(sequence, columns=['value'])
            sequence = pd.DataFrame(
                self.scaler.transform(sequence_df),
                columns=sequence_df.columns
            ).values.flatten()
            
            # Convert to tensor and reshape: (1, seq_length, 1)
            sequence = torch.FloatTensor(sequence).view(1, -1, 1).to(self.device)
            
            # Get prediction
            prediction = self.model(sequence)
            # Convert back to numpy and inverse transform
            prediction = prediction.cpu().numpy()
            prediction_df = pd.DataFrame(prediction.reshape(-1, 1), columns=['value'])
            prediction = self.scaler.inverse_transform(prediction_df)
            
            return prediction[0, 0]

    def evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.model.eval()
        criterion = nn.MSELoss()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.view(batch_X.size(0), -1, 1).to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        return avg_loss 