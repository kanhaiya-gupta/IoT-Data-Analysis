import pandas as pd
import numpy as np
from lstm_forecasting import LSTMForecasting
import matplotlib.pyplot as plt
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_root, 'outputs', 'lstm')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def create_sample_data():
    """Create sample time series data with trend and seasonality"""
    np.random.seed(42)
    time = np.arange(0, 1000)
    trend = 0.1 * time
    seasonality = 50 * np.sin(2 * np.pi * time / 100)
    noise = np.random.normal(0, 10, len(time))
    
    data = trend + seasonality + noise
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=len(time), freq='h'),
        'value': data
    })
    return df

def main():
    # Create sample data
    logger.info("Creating sample data...")
    df = create_sample_data()
    
    # Initialize forecaster
    logger.info("Initializing LSTM forecaster...")
    forecaster = LSTMForecasting(
        seq_length=24,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        num_epochs=50,
        batch_size=32
    )
    
    # Prepare data
    logger.info("Preparing data...")
    train_loader, test_loader = forecaster.prepare_data(df, 'value')
    
    # Train model
    logger.info("Training model...")
    losses = forecaster.train(train_loader)
    
    # Evaluate model
    logger.info("Evaluating model...")
    test_loss = forecaster.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    logger.info("Making predictions...")
    last_sequence = df['value'].values[-24:].reshape(-1, 1)
    predictions = []
    
    for _ in range(24):  # Predict next 24 hours
        pred = forecaster.predict(last_sequence)
        predictions.append(pred)
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = pred
    
    # Plot results
    logger.info("Plotting results...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['value'].values[-48:], label='Actual', color='blue')
    plt.plot(range(24, 48), predictions, label='Predicted', color='red', linestyle='--')
    plt.legend()
    plt.title('LSTM Forecasting Results')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'forecast_results.png'))
    plt.close()
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    logger.info(f"Forecasting completed. Results saved as '{os.path.join(output_dir, 'forecast_results.png')}'")
    logger.info(f"Training loss plot saved as '{os.path.join(output_dir, 'training_loss.png')}'")

if __name__ == "__main__":
    main() 