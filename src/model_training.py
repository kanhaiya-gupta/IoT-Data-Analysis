import pandas as pd
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple, List
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import paho.mqtt.subscribe as subscribe

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_training.log"),
        logging.StreamHandler()
    ]
)

class IoTModel:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_columns = None
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
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest") -> Optional[Pipeline]:
        """Train a model with hyperparameter tuning."""
        try:
            logging.info(f"Starting model training with {model_type}...")
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logging.info(f"Data split: Train size: {X_train.shape}, Test size: {X_test.shape}")
            
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
            
            # Evaluate model
            logging.info("Evaluating model on test set...")
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1]
            
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))
            logging.info("\nConfusion Matrix:")
            logging.info(confusion_matrix(y_test, y_pred))
            
            # Plot ROC curve
            logging.info("Generating ROC curve...")
            self.plot_roc_curve(y_test, y_proba)
            
            # Plot feature importance for Random Forest
            if model_type == "random_forest":
                logging.info("Generating feature importance plot...")
                self.plot_feature_importance(X.columns)
            
            # Save model
            logging.info("Saving model and metadata...")
            self.save_model()
            
            logging.info("Model training completed successfully")
            return self.model
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return None
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray):
        """Plot ROC curve for model evaluation."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(self.model_dir / "roc_curve.png")
        plt.close()
        logging.info(f"ROC curve saved with AUC: {roc_auc:.3f}")
    
    def plot_feature_importance(self, feature_names: List[str]):
        """Plot feature importance for Random Forest model."""
        if isinstance(self.model.named_steps["model"], RandomForestClassifier):
            importances = self.model.named_steps["model"].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importances')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.model_dir / "feature_importance.png")
            plt.close()
            
            # Log feature importances
            logging.info("\nFeature Importances:")
            for i, idx in enumerate(indices):
                logging.info(f"{feature_names[idx]}: {importances[idx]:.4f}")
    
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