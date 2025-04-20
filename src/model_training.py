import pandas as pd
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple, List
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import paho.mqtt.subscribe as subscribe
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_training.log"),
        logging.StreamHandler()
    ]
)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        return x

class IoTModel:
    def __init__(self, model_dir: str = "models", output_dir: str = "outputs/training"):
        """Initialize the IoT model with directories for model and output storage."""
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.model = None
        self.feature_columns = None
        self.metadata = {}
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Initialized IoTModel with model directory: {model_dir}")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from the data."""
        try:
            logging.info("Creating time-based features...")
            # Time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            logging.info("Creating environmental interaction features...")
            # Environmental interaction features
            df['temp_humidity'] = df['temperature'] * df['humidity']
            df['radiation_pressure'] = df['radiation'] * df['pressure']
            
            logging.info("Creating rolling window features...")
            # Rolling window features
            window = 6  # 1-hour window for 10-minute data
            df['rolling_avg_vehicles'] = (df['light_vehicles'] + df['heavy_vehicles']).rolling(window).mean()
            df['rolling_std_vehicles'] = (df['light_vehicles'] + df['heavy_vehicles']).rolling(window).std()
            
            # Fill NaN values from rolling windows
            df = df.ffill()
            
            logging.info("Feature creation completed successfully")
            return df
        except Exception as e:
            logging.error(f"Error creating features: {e}")
            raise
    
    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable based on traffic patterns."""
        try:
            logging.info("Creating target variable...")
            # Calculate total vehicles
            total_vehicles = df['light_vehicles'] + df['heavy_vehicles']
            
            # Use dynamic threshold based on historical data
            threshold = total_vehicles.quantile(0.75)  # High traffic is above 75th percentile
            logging.info(f"Target threshold (75th percentile): {threshold:.2f}")
            
            # Create target variable
            y = (total_vehicles > threshold).astype(int)
            
            logging.info(f"Target distribution:\n{y.value_counts(normalize=True)}")
            return y
        except Exception as e:
            logging.error(f"Error creating target: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        try:
            logging.info("Starting data preparation...")
            # Create features
            df = self.create_features(df)
            
            # Create target
            y = self.create_target(df)
            
            # Select features for training
            feature_columns = [
                'precipitation', 'humidity', 'radiation', 'sunshine', 'pressure', 'temperature',
                'hour', 'day_of_week', 'is_weekend',
                'temp_humidity', 'radiation_pressure',
                'rolling_avg_vehicles', 'rolling_std_vehicles'
            ]
            
            logging.info(f"Selected {len(feature_columns)} features for training")
            logging.info(f"Features: {feature_columns}")
            
            X = df[feature_columns]
            self.feature_columns = feature_columns
            
            logging.info(f"Data preparation completed. X shape: {X.shape}, y shape: {y.shape}")
            return X, y
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise
    
    def plot_roc_curve(self, y_true_train: np.ndarray, y_proba_train: np.ndarray, 
                      y_true_test: np.ndarray, y_proba_test: np.ndarray):
        """Plot ROC curves for both training and test sets."""
        try:
            # Calculate ROC curves for both sets
            fpr_train, tpr_train, _ = roc_curve(y_true_train, y_proba_train)
            fpr_test, tpr_test, _ = roc_curve(y_true_test, y_proba_test)
            
            # Calculate AUC for both sets
            roc_auc_train = auc(fpr_train, tpr_train)
            roc_auc_test = auc(fpr_test, tpr_test)
            
            # Create the plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_train, tpr_train, color='blue', lw=2, 
                    label=f'Train ROC (AUC = {roc_auc_train:.2f})')
            plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, 
                    label=f'Test ROC (AUC = {roc_auc_test:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curves')
            plt.legend(loc="lower right")
            
            # Save plot with error handling
            roc_path = self.output_dir / "roc_curve.png"
            plt.savefig(roc_path)
            plt.close()
            logging.info(f"ROC curve saved successfully to: {roc_path}")
            
            # Log AUC scores
            logging.info(f"Training AUC: {roc_auc_train:.3f}")
            logging.info(f"Test AUC: {roc_auc_test:.3f}")
            
            # Check for overfitting
            auc_diff = abs(roc_auc_train - roc_auc_test)
            if auc_diff > 0.1:
                logging.warning(f"Potential overfitting detected: AUC difference = {auc_diff:.3f}")
        except Exception as e:
            logging.error(f"Error saving ROC curve: {e}")
            raise
    
    def plot_feature_importance(self, feature_names: List[str]):
        """Plot feature importance for Random Forest model."""
        try:
            if isinstance(self.model.named_steps["model"], RandomForestClassifier):
                importances = self.model.named_steps["model"].feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title('Feature Importances')
                plt.bar(range(len(importances)), importances[indices], align='center')
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
                plt.tight_layout()
                
                # Save plot with error handling
                importance_path = self.output_dir / "feature_importance.png"
                plt.savefig(importance_path)
                plt.close()
                logging.info(f"Feature importance plot saved successfully to: {importance_path}")
                
                # Log feature importances
                logging.info("\nFeature Importances:")
                for i, idx in enumerate(indices):
                    logging.info(f"{feature_names[idx]}: {importances[idx]:.4f}")
            else:
                logging.warning("Feature importance plot skipped: Not a Random Forest model")
        except Exception as e:
            logging.error(f"Error saving feature importance plot: {e}")
            raise
    
    def plot_training_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray, 
                            y_test: np.ndarray, y_pred_test: np.ndarray):
        """Plot training and test metrics for model evaluation."""
        try:
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_f1 = f1_score(y_train, y_pred_train)
            test_f1 = f1_score(y_test, y_pred_test)
            train_precision = precision_score(y_train, y_pred_train)
            test_precision = precision_score(y_test, y_pred_test)
            train_recall = recall_score(y_train, y_pred_train)
            test_recall = recall_score(y_test, y_pred_test)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Metrics', fontsize=16)
            
            # Plot metrics
            metrics = [
                ('Accuracy', train_accuracy, test_accuracy),
                ('F1 Score', train_f1, test_f1),
                ('Precision', train_precision, test_precision),
                ('Recall', train_recall, test_recall)
            ]
            
            for idx, (title, train_val, test_val) in enumerate(metrics):
                row = idx // 2
                col = idx % 2
                axes[row, col].bar(['Train', 'Test'], [train_val, test_val], 
                                 color=['blue', 'orange'])
                axes[row, col].set_title(title)
                axes[row, col].set_ylim(0, 1)
                for i, v in enumerate([train_val, test_val]):
                    axes[row, col].text(i, v + 0.02, f'{v:.3f}', ha='center')
            
            plt.tight_layout()
            
            # Save plot with error handling
            metrics_path = self.output_dir / "training_metrics.png"
            plt.savefig(metrics_path)
            plt.close()
            logging.info(f"Training metrics plot saved successfully to: {metrics_path}")
            
            # Log metrics
            logging.info("\nTraining Metrics:")
            logging.info(f"Accuracy: {train_accuracy:.3f}")
            logging.info(f"F1 Score: {train_f1:.3f}")
            logging.info(f"Precision: {train_precision:.3f}")
            logging.info(f"Recall: {train_recall:.3f}")
            
            logging.info("\nTest Metrics:")
            logging.info(f"Accuracy: {test_accuracy:.3f}")
            logging.info(f"F1 Score: {test_f1:.3f}")
            logging.info(f"Precision: {test_precision:.3f}")
            logging.info(f"Recall: {test_recall:.3f}")
            
            # Check for overfitting
            accuracy_diff = abs(train_accuracy - test_accuracy)
            if accuracy_diff > 0.1:
                logging.warning(f"Potential overfitting detected: Accuracy difference = {accuracy_diff:.3f}")
        except Exception as e:
            logging.error(f"Error saving training metrics plot: {e}")
            raise
    
    def plot_training_history(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """Plot training history for neural network."""
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot loss
            ax1.plot(train_losses, label='Training Loss')
            ax1.plot(val_losses, label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Plot accuracy
            ax2.plot(train_accuracies, label='Training Accuracy')
            ax2.plot(val_accuracies, label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save plot
            history_path = self.output_dir / "training_history.png"
            plt.savefig(history_path)
            plt.close()
            logging.info(f"Training history plot saved successfully to: {history_path}")
            
            # Log final metrics
            logging.info("\nFinal Training Metrics:")
            logging.info(f"Training Loss: {train_losses[-1]:.4f}")
            logging.info(f"Training Accuracy: {train_accuracies[-1]:.4f}")
            logging.info(f"Validation Loss: {val_losses[-1]:.4f}")
            logging.info(f"Validation Accuracy: {val_accuracies[-1]:.4f}")
            
        except Exception as e:
            logging.error(f"Error saving training history plot: {e}")
            raise
    
    def plot_classifier_performance(self, y_train: np.ndarray, y_pred_train: np.ndarray, 
                                  y_test: np.ndarray, y_pred_test: np.ndarray):
        """Plot detailed classifier performance including confusion matrices and classification reports."""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Classifier Performance Analysis', fontsize=16)
            
            # Plot confusion matrices
            for idx, (y_true, y_pred, title) in enumerate([
                (y_train, y_pred_train, 'Training Set'),
                (y_test, y_pred_test, 'Test Set')
            ]):
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, idx])
                axes[0, idx].set_title(f'Confusion Matrix - {title}')
                axes[0, idx].set_xlabel('Predicted')
                axes[0, idx].set_ylabel('Actual')
            
            # Calculate metrics
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            train_metrics = [
                accuracy_score(y_train, y_pred_train),
                precision_score(y_train, y_pred_train),
                recall_score(y_train, y_pred_train),
                f1_score(y_train, y_pred_train)
            ]
            test_metrics = [
                accuracy_score(y_test, y_pred_test),
                precision_score(y_test, y_pred_test),
                recall_score(y_test, y_pred_test),
                f1_score(y_test, y_pred_test)
            ]
            
            # Plot metrics comparison
            x = np.arange(len(metrics))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, train_metrics, width, label='Train')
            axes[1, 0].bar(x + width/2, test_metrics, width, label='Test')
            axes[1, 0].set_title('Performance Metrics Comparison')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(metrics)
            axes[1, 0].legend()
            axes[1, 0].set_ylim(0, 1)
            
            # Add metric values on top of bars
            for i, (train_val, test_val) in enumerate(zip(train_metrics, test_metrics)):
                axes[1, 0].text(i - width/2, train_val + 0.02, f'{train_val:.3f}', ha='center')
                axes[1, 0].text(i + width/2, test_val + 0.02, f'{test_val:.3f}', ha='center')
            
            # Classification reports
            train_report = classification_report(y_train, y_pred_train, output_dict=True)
            test_report = classification_report(y_test, y_pred_test, output_dict=True)
            
            # Plot classification reports
            report_text = f"Training Set Classification Report:\n\n{classification_report(y_train, y_pred_train)}\n\n"
            report_text += f"Test Set Classification Report:\n\n{classification_report(y_test, y_pred_test)}"
            
            axes[1, 1].text(0.1, 0.5, report_text, fontfamily='monospace', fontsize=10)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            classifier_path = self.output_dir / "classifier_performance.png"
            plt.savefig(classifier_path)
            plt.close()
            logging.info(f"Classifier performance plot saved successfully to: {classifier_path}")
            
            # Log classification reports
            logging.info("\nTraining Set Classification Report:")
            logging.info(classification_report(y_train, y_pred_train))
            logging.info("\nTest Set Classification Report:")
            logging.info(classification_report(y_test, y_pred_test))
            
        except Exception as e:
            logging.error(f"Error saving classifier performance plot: {e}")
            raise
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest") -> Optional[Pipeline]:
        """Train a model with hyperparameter tuning."""
        try:
            logging.info(f"Starting model training with {model_type}...")
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logging.info(f"Data split: Train size: {X_train.shape}, Test size: {X_test.shape}")
            
            if model_type == "neural_network":
                # Scale the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Convert to PyTorch tensors
                X_train_tensor = torch.FloatTensor(X_train_scaled)
                y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
                
                # Create datasets and dataloaders
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                # Initialize model
                model = NeuralNetwork(input_size=X_train.shape[1])
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters())
                
                # Training loop
                num_epochs = 100
                train_losses = []
                val_losses = []
                train_accuracies = []
                val_accuracies = []
                
                for epoch in range(num_epochs):
                    # Training
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0
                    
                    for inputs, labels in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        predicted = (outputs > 0.5).float()
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()
                    
                    train_losses.append(train_loss / len(train_loader))
                    train_accuracies.append(train_correct / train_total)
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                            
                            predicted = (outputs > 0.5).float()
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()
                    
                    val_losses.append(val_loss / len(test_loader))
                    val_accuracies.append(val_correct / val_total)
                    
                    # Early stopping
                    if epoch > 10 and val_losses[-1] > val_losses[-2]:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break
                    
                    if (epoch + 1) % 10 == 0:
                        logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
                                   f'Train Loss: {train_losses[-1]:.4f}, '
                                   f'Train Acc: {train_accuracies[-1]:.4f}, '
                                   f'Val Loss: {val_losses[-1]:.4f}, '
                                   f'Val Acc: {val_accuracies[-1]:.4f}')
                
                # Plot training history
                self.plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
                
                # Get predictions
                model.eval()
                with torch.no_grad():
                    y_pred_train = (model(X_train_tensor) > 0.5).float().numpy()
                    y_pred_test = (model(X_test_tensor) > 0.5).float().numpy()
                    y_proba_train = model(X_train_tensor).numpy()
                    y_proba_test = model(X_test_tensor).numpy()
                
                # Plot ROC curves
                self.plot_roc_curve(y_train, y_proba_train, y_test, y_proba_test)
                
                # Plot training metrics
                self.plot_training_metrics(y_train, y_pred_train, y_test, y_pred_test)
                
                # Convert predictions to binary
                y_pred_train = (y_pred_train > 0.5).astype(int)
                y_pred_test = (y_pred_test > 0.5).astype(int)
                
                # Plot classifier performance
                self.plot_classifier_performance(y_train, y_pred_train, y_test, y_pred_test)
                
                # Save model
                self.model = model
                self.save_model()
                
                return model
                
            else:
                # Create pipeline
                if model_type == "logistic":
                    logging.info("Setting up Logistic Regression pipeline...")
                    pipeline = Pipeline([
                        ("scale", StandardScaler()),
                        ("model", LogisticRegression())
                    ])
                    param_grid = {
                        "model__C": [0.1, 1, 10],
                        "model__penalty": ["l1", "l2"],
                        "model__solver": ["liblinear"]
                    }
                elif model_type == "random_forest":
                    logging.info("Setting up Random Forest pipeline...")
                    pipeline = Pipeline([
                        ("scale", StandardScaler()),
                        ("model", RandomForestClassifier())
                    ])
                    param_grid = {
                        "model__n_estimators": [100, 200, 300],
                        "model__max_depth": [None, 10, 20, 30],
                        "model__min_samples_split": [2, 5, 10],
                        "model__min_samples_leaf": [1, 2, 4]
                    }
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                logging.info("Starting grid search for hyperparameter tuning...")
                # Grid search for hyperparameter tuning
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=5,
                    scoring="f1",
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit model
                logging.info("Fitting model with grid search...")
                grid_search.fit(X_train, y_train)
                
                # Get best model
                self.model = grid_search.best_estimator_
                logging.info(f"Best parameters found: {grid_search.best_params_}")
                
                # Cross-validation scores
                cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring="f1")
                logging.info(f"\nCross-validation scores: {cv_scores}")
                logging.info(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
                # Get predictions for both sets
                y_pred_train = self.model.predict(X_train)
                y_proba_train = self.model.predict_proba(X_train)[:, 1]
                y_pred_test = self.model.predict(X_test)
                y_proba_test = self.model.predict_proba(X_test)[:, 1]
                
                # Plot ROC curves for both sets
                logging.info("Generating ROC curves...")
                self.plot_roc_curve(y_train, y_proba_train, y_test, y_proba_test)
                
                # Plot training metrics
                logging.info("Generating training metrics...")
                self.plot_training_metrics(y_train, y_pred_train, y_test, y_pred_test)
                
                # Plot feature importance for Random Forest
                if model_type == "random_forest":
                    logging.info("Generating feature importance plot...")
                    self.plot_feature_importance(X.columns)
                
                # Plot classifier performance
                self.plot_classifier_performance(y_train, y_pred_train, y_test, y_pred_test)
                
                # Save model
                logging.info("Saving model and metadata...")
                self.save_model()
                
                logging.info("Model training completed successfully")
                return self.model
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return None
    
    def save_model(self):
        """Save trained model and metadata."""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            # Save model
            model_path = self.model_dir / "iot_model.pkl"
            with model_path.open("wb") as f:
                pickle.dump(self.model, f)
            
            # Save metadata
            metadata = {
                "feature_columns": self.feature_columns,
                "timestamp": datetime.now().isoformat(),
                "model_type": type(self.model.named_steps["model"]).__name__,
                "model_params": self.model.get_params()
            }
            
            metadata_path = self.model_dir / "model_metadata.json"
            with metadata_path.open("w") as f:
                json.dump(metadata, f)
            
            logging.info(f"Model and metadata saved to {self.model_dir}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self) -> bool:
        """Load trained model and metadata."""
        try:
            # Load model
            model_path = self.model_dir / "iot_model.pkl"
            with model_path.open("rb") as f:
                self.model = pickle.load(f)
            
            # Load metadata
            metadata_path = self.model_dir / "model_metadata.json"
            with metadata_path.open("r") as f:
                metadata = json.load(f)
                self.feature_columns = metadata["feature_columns"]
            
            logging.info(f"Model and metadata loaded from {self.model_dir}")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def predict(self, data: Dict[str, Any]) -> Optional[Any]:
        """Make prediction on new data."""
        try:
            if self.model is None:
                raise ValueError("No model loaded")
            
            # Convert data to DataFrame
            df = pd.DataFrame([data])
            
            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Make prediction
            prediction = self.model.predict(df[self.feature_columns])
            probabilities = self.model.predict_proba(df[self.feature_columns])
            
            return {
                "prediction": prediction[0],
                "probabilities": probabilities[0].tolist()
            }
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            return None

def main():
    """Main function to train and evaluate model."""
    try:
        logging.info("Starting model training process...")
        # Load and prepare data
        from data_acquisition import load_environmental_data, load_traffic_data
        
        logging.info("Loading data files...")
        # Load data
        df_env = load_environmental_data()
        traffic_data = load_traffic_data()
        
        if df_env is not None and all(df is not None for df in traffic_data.values()):
            logging.info("Data loaded successfully")
            # Combine data
            from data_processing import combine_iot_data
            combined_df = combine_iot_data(df_env, traffic_data)
            logging.info(f"Combined data shape: {combined_df.shape}")
            
            # Initialize and train model
            model = IoTModel()
            X, y = model.prepare_data(combined_df)
            model.train_model(X, y, model_type="random_forest")
        else:
            logging.error("Failed to load one or more data files")
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()