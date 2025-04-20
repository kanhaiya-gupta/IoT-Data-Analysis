import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/advanced_analysis.log'),
        logging.StreamHandler()
    ]
)

class AdvancedAnalyzer:
    def __init__(self, output_dir: str = "outputs/advanced_analysis"):
        """Initialize the analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Set up consistent plot style."""
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (15, 8)
        plt.rcParams['font.size'] = 12
    
    def check_stationarity(self, series: pd.Series, window: int = 12) -> Dict[str, float]:
        """Check time series stationarity using Augmented Dickey-Fuller test."""
        result = adfuller(series.dropna())
        return {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }
    
    def detect_anomalies(self, df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest."""
        try:
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols])
            
            # Fit Isolation Forest
            clf = IsolationForest(contamination=contamination, random_state=42)
            predictions = clf.fit_predict(scaled_data)
            
            # Add anomaly column
            df['is_anomaly'] = predictions == -1
            
            # Calculate anomaly scores
            df['anomaly_score'] = -clf.score_samples(scaled_data)
            
            return df
        except Exception as e:
            logging.error(f"Error in anomaly detection: {e}")
            return df
    
    def analyze_seasonality(self, df: pd.DataFrame, column: str, periods: List[int] = [24, 168]) -> Dict[str, Any]:
        """Analyze multiple seasonal patterns in time series data."""
        try:
            results = {}
            series = df[column].ffill()
            
            for period in periods:
                if len(series) >= 2 * period:
                    decomp = sm.tsa.seasonal_decompose(series, period=period)
                    results[f'period_{period}'] = {
                        'trend': decomp.trend,
                        'seasonal': decomp.seasonal,
                        'residual': decomp.resid
                    }
            
            return results
        except Exception as e:
            logging.error(f"Error in seasonality analysis: {e}")
            return {}
    
    def plot_advanced_time_series(self, df: pd.DataFrame, column: str):
        """Create advanced time series visualization with multiple components."""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(4, 1)
            
            # Original series
            ax1 = fig.add_subplot(gs[0, 0])
            df[column].plot(ax=ax1)
            ax1.set_title(f'Original {column} Time Series')
            
            # Rolling statistics
            ax2 = fig.add_subplot(gs[1, 0])
            rolling_mean = df[column].rolling(window=24).mean()
            rolling_std = df[column].rolling(window=24).std()
            df[column].plot(ax=ax2, label='Original')
            rolling_mean.plot(ax=ax2, label='Rolling Mean')
            rolling_std.plot(ax=ax2, label='Rolling Std')
            ax2.legend()
            ax2.set_title('Rolling Statistics')
            
            # Seasonal decomposition
            seasonal_results = self.analyze_seasonality(df, column)
            if seasonal_results:
                ax3 = fig.add_subplot(gs[2, 0])
                seasonal_results['period_24']['trend'].plot(ax=ax3, label='Trend')
                seasonal_results['period_24']['seasonal'].plot(ax=ax3, label='Seasonal')
                ax3.legend()
                ax3.set_title('Trend and Seasonal Components')
            
            # Anomaly detection
            df_with_anomalies = self.detect_anomalies(df[[column]])
            ax4 = fig.add_subplot(gs[3, 0])
            df[column].plot(ax=ax4, label='Original')
            anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']]
            ax4.scatter(anomalies.index, anomalies[column], color='red', label='Anomalies')
            ax4.legend()
            ax4.set_title('Anomaly Detection')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'advanced_time_series_{column}.png', dpi=300)
            plt.close()
            logging.info(f"Generated advanced time series plot for {column}")
        except Exception as e:
            logging.error(f"Error in advanced time series plotting: {e}")
    
    def analyze_cross_correlations(self, df: pd.DataFrame, target_col: str, 
                                 max_lags: int = 24) -> Dict[str, List[float]]:
        """Analyze cross-correlations between variables with different lags."""
        try:
            correlations = {}
            for col in df.columns:
                if col != target_col and pd.api.types.is_numeric_dtype(df[col]):
                    corr_values = []
                    for lag in range(-max_lags, max_lags + 1):
                        if lag < 0:
                            corr = df[target_col].corr(df[col].shift(-lag))
                        else:
                            corr = df[col].corr(df[target_col].shift(lag))
                        corr_values.append(corr)
                    correlations[col] = corr_values
            return correlations
        except Exception as e:
            logging.error(f"Error in cross-correlation analysis: {e}")
            return {}
    
    def plot_cross_correlations(self, correlations: Dict[str, List[float]], max_lags: int = 24):
        """Plot cross-correlation analysis results."""
        try:
            fig, ax = plt.subplots(figsize=(15, 8))
            lags = range(-max_lags, max_lags + 1)
            
            for col, corr_values in correlations.items():
                ax.plot(lags, corr_values, label=col)
            
            ax.axhline(y=0, color='black', linestyle='--')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Correlation')
            ax.set_title('Cross-Correlation Analysis')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'cross_correlations.png', dpi=300)
            plt.close()
            logging.info("Generated cross-correlation plot")
        except Exception as e:
            logging.error(f"Error plotting cross-correlations: {e}")
    
    def analyze_all(self, df: pd.DataFrame, target_cols: List[str] = None):
        """Perform comprehensive analysis of the dataset."""
        try:
            if target_cols is None:
                target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Analyze each target column
            for col in target_cols:
                # Stationarity check
                stationarity = self.check_stationarity(df[col])
                logging.info(f"\nStationarity analysis for {col}:")
                logging.info(f"ADF Statistic: {stationarity['ADF Statistic']:.4f}")
                logging.info(f"p-value: {stationarity['p-value']:.4f}")
                
                # Advanced time series analysis
                self.plot_advanced_time_series(df, col)
                
                # Cross-correlation analysis
                correlations = self.analyze_cross_correlations(df, col)
                self.plot_cross_correlations(correlations)
                
                # Anomaly detection
                df_with_anomalies = self.detect_anomalies(df[[col]])
                anomaly_count = df_with_anomalies['is_anomaly'].sum()
                logging.info(f"Detected {anomaly_count} anomalies in {col}")
            
            logging.info("Advanced analysis completed successfully")
        except Exception as e:
            logging.error(f"Error in comprehensive analysis: {e}")

def main():
    """Main function to demonstrate advanced analysis."""
    try:
        # Initialize analyzer
        analyzer = AdvancedAnalyzer()
        
        # Load data (assuming data is already loaded)
        from data_acquisition import load_environmental_data, load_traffic_data
        
        # Load environmental data
        env_df = load_environmental_data()
        if env_df is not None:
            logging.info("Analyzing environmental data...")
            analyzer.analyze_all(env_df)
        
        # Load traffic data
        traffic_data = load_traffic_data()
        for vehicle_type, df in traffic_data.items():
            if df is not None:
                logging.info(f"Analyzing {vehicle_type} data...")
                analyzer.analyze_all(df)
        
        logging.info("Advanced analysis completed")
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 