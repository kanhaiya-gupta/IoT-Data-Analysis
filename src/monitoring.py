import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import time
from pathlib import Path
import json
import os
import ijson  # For streaming JSON parsing
import aiohttp
import asyncio
import websockets

# Get the absolute path of the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Create necessary directories
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "monitoring"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Monitoring system initialized")

class DataMonitor:
    def __init__(self):
        self.data_window = []
        
    def update_data_window(self, new_data: Dict[str, Any]):
        """Update the data window with new data."""
        try:
            # Convert timestamp to datetime if it's a string
            if isinstance(new_data.get('timestamp'), str):
                new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
            
            self.data_window.append(new_data)
                
        except Exception as e:
            logger.error(f"Error updating data window: {e}")
        
    def analyze_data(self) -> Dict[str, Any]:
        """Analyze the current data window."""
        try:
            if not self.data_window:
                return {
                    'statistics': {
                        'timestamp': datetime.now().isoformat(),
                        'value': 0
                    },
                    'anomalies': [],
                    'data_quality': {
                        'timestamp': datetime.now().isoformat(),
                        'completeness': 0.0
                    }
                }
            
            # Get the latest data point
            latest_data = self.data_window[-1]
            
            # Prepare statistics data
            stats = {
                'timestamp': latest_data['timestamp'].isoformat(),
                'value': float(latest_data['value'])
            }
            
            return {
                'statistics': stats,
                'anomalies': [],
                'data_quality': {
                    'timestamp': latest_data['timestamp'].isoformat(),
                    'completeness': 1.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_data: {e}")
            return {
                'statistics': {
                    'timestamp': datetime.now().isoformat(),
                    'value': 0
                },
                'anomalies': [],
                'data_quality': {
                    'timestamp': datetime.now().isoformat(),
                    'completeness': 0.0
                }
            }
    
    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies in the data."""
        anomalies = []
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for column in numeric_cols:
                if column == 'timestamp':
                    continue
                    
                # Calculate z-scores
                mean = df[column].mean()
                std = df[column].std()
                if pd.isna(mean) or pd.isna(std) or std == 0:
                    continue
                    
                z_scores = np.abs((df[column] - mean) / std)
                
                # Find anomalies (z-score > 3)
                anomaly_indices = np.where(z_scores > 3)[0]
                
                for idx in anomaly_indices:
                    anomalies.append({
                        'timestamp': df.iloc[idx]['timestamp'].isoformat(),
                        'variable': column,
                        'value': float(df[column].iloc[idx]),
                        'z_score': float(z_scores[idx])
                    })
                    
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            
        return anomalies
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of the data."""
        quality = {
            'timestamp': datetime.now().isoformat(),
            'completeness': {},
            'timeliness': {}
        }
        
        try:
            # Check completeness
            for column in df.columns:
                if column == 'timestamp':
                    continue
                missing_ratio = df[column].isnull().mean()
                quality['completeness'][column] = float(1 - missing_ratio)
            
            # Check timeliness
            if 'timestamp' in df.columns:
                time_diffs = df['timestamp'].diff().dt.total_seconds()
                quality['timeliness'] = {
                    'mean_interval': float(time_diffs.mean()),
                    'max_interval': float(time_diffs.max())
                }
                
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            
        return quality
    
    def generate_alerts(self, stats: Dict[str, Any], 
                       anomalies: List[Dict[str, Any]], 
                       quality: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on monitoring metrics."""
        alerts = []
        try:
            thresholds = self.config.get('alert_thresholds', {})
            
            # Check data quality alerts
            for column, completeness in quality['completeness'].items():
                if completeness < thresholds.get('completeness_threshold', 0.95):
                    alerts.append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'data_quality',
                        'metric': 'completeness',
                        'variable': column,
                        'value': float(completeness)
                    })
            
            # Check anomaly alerts
            if len(anomalies) > 5:  # Too many anomalies
                alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'anomaly_rate',
                    'value': len(anomalies)
                })
                
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            
        return alerts
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save monitoring metrics to files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save statistics
            if metrics.get('statistics'):
                stats_file = self.output_dir / f"statistics_{timestamp}.json"
                with open(stats_file, 'w') as f:
                    json.dump(metrics['statistics'], f, indent=2)
                logger.info(f"Saved statistics to {stats_file}")
            
            # Save anomalies
            if metrics.get('anomalies'):
                anomalies_file = self.output_dir / f"anomalies_{timestamp}.json"
                with open(anomalies_file, 'w') as f:
                    json.dump(metrics['anomalies'], f, indent=2)
                logger.info(f"Saved anomalies to {anomalies_file}")
            
            # Save data quality
            if metrics.get('data_quality'):
                quality_file = self.output_dir / f"quality_{timestamp}.json"
                with open(quality_file, 'w') as f:
                    json.dump(metrics['data_quality'], f, indent=2)
                logger.info(f"Saved data quality to {quality_file}")
            
            # Save alerts
            if metrics.get('alerts'):
                alerts_file = self.output_dir / f"alerts_{timestamp}.json"
                with open(alerts_file, 'w') as f:
                    json.dump(metrics['alerts'], f, indent=2)
                logger.info(f"Saved alerts to {alerts_file}")
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def process_file(self, file_path: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Process a data file and return metrics."""
        try:
            with open(file_path, 'rb') as f:
                # Use ijson for memory-efficient parsing
                parser = ijson.parse(f)
                for prefix, event, value in parser:
                    if event == 'map_key':
                        continue
                        
                    if isinstance(value, dict):
                        value['data_type'] = data_type
                        self.data_window.append(value)
                        
                        # Only analyze after collecting enough data
                        if len(self.data_window) >= self.max_window_size:
                            metrics = self.analyze_data()
                            self.save_metrics(metrics)
                            return metrics
                            
            return None
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def check_anomalies(self) -> Dict[str, Any]:
        """Check for anomalies in the current data window."""
        current_time = time.time()
        if current_time - self.last_check_time < self.sampling_interval:
            return {}
        
        self.last_check_time = current_time
        logger.info("Checking for anomalies...")
        
        if len(self.data_window) < self.max_window_size:
            logger.warning(f"Insufficient data points for anomaly detection. Current: {len(self.data_window)}, Required: {self.max_window_size}")
            return {}
        
        # Convert data window to DataFrame for analysis
        df = pd.DataFrame(self.data_window)
        anomalies = {}
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            anomalies['missing_values'] = missing_values[missing_values > 0].to_dict()
            logger.warning(f"Missing values detected: {anomalies['missing_values']}")
        
        # Check for outliers using IQR method
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
            if not outliers.empty:
                anomalies[f'outliers_{column}'] = outliers[column].to_dict()
                logger.warning(f"Outliers detected in {column}: {len(outliers)} points")
        
        # Check for sudden changes
        for column in df.select_dtypes(include=[np.number]).columns:
            changes = df[column].diff().abs()
            large_changes = changes[changes > changes.mean() + 3 * changes.std()]
            if not large_changes.empty:
                anomalies[f'sudden_changes_{column}'] = large_changes.to_dict()
                logger.warning(f"Sudden changes detected in {column}: {len(large_changes)} points")
        
        if anomalies:
            logger.info(f"Anomalies detected: {list(anomalies.keys())}")
        else:
            logger.info("No anomalies detected in current window")
        
        return anomalies

    async def send_monitoring_data(self, websocket):
        """Send monitoring data through WebSocket."""
        try:
            while True:
                anomalies = self.check_anomalies()
                if anomalies:
                    data = {
                        'timestamp': datetime.now().isoformat(),
                        'anomalies': anomalies,
                        'data_window_size': len(self.data_window)
                    }
                    await websocket.send(json.dumps(data))
                    logger.debug(f"Sent monitoring data: {data}")
                await asyncio.sleep(self.sampling_interval)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error sending monitoring data: {str(e)}")

    async def process_data_point(self, data_point: Dict[str, Any]):
        """Process a single data point."""
        try:
            self.data_window.append(data_point)
            logger.debug(f"Processed data point: {data_point}")
        except Exception as e:
            logger.error(f"Error processing data point: {str(e)}")

def load_large_json(file_path: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """Load large JSON file using streaming parser."""
    try:
        with open(file_path, 'rb') as f:
            # Use ijson to parse the file in a streaming fashion
            parser = ijson.parse(f)
            
            current_chunk = []
            current_object = {}
            current_key = None
            
            for prefix, event, value in parser:
                if prefix == 'item' and event == 'start_map':
                    current_object = {}
                elif prefix == 'item' and event == 'end_map':
                    current_chunk.append(current_object)
                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []
                elif event == 'map_key':
                    current_key = value
                elif event in ['string', 'number', 'boolean']:
                    current_object[current_key] = value
                
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        yield []

async def process_data_point(monitor: DataMonitor, point: Dict[str, Any], data_type: str):
    """Process a single data point."""
    try:
        # Convert timestamp to datetime if it's not already
        if isinstance(point['timestamp'], (int, float, str)):
            point['timestamp'] = pd.to_datetime(point['timestamp'])
        
        # Add data type to the point
        point['data_type'] = data_type
        
        # Ensure numeric values
        if 'value' in point:
            point['value'] = float(point['value'])
        
        # For environmental data, ensure all values are numeric
        if data_type == 'environmental':
            for key in ['precipitation', 'humidity', 'radiation', 'sunshine', 'pressure', 'temperature']:
                if key in point:
                    point[key] = float(point[key])
        
        # Update monitor with the data point
        await monitor.process_data_point(point)
        
        # Analyze data and send updates
        await monitor.analyze_data()
        
    except Exception as e:
        logging.error(f"Error processing data point: {e}")
        logging.error(f"Data point: {point}")

async def main():
    """Main function to demonstrate the monitoring system with existing data."""
    # Configuration
    config = {
        'max_window_size': 1000,
        'sampling_interval': 1,  # Reduced to 1 second for more frequent updates
        'output_dir': 'outputs/monitoring',
        'api_url': 'ws://localhost:8000/ws',  # Changed to ws:// for WebSocket
        'alert_thresholds': {
            'completeness_threshold': 0.95,
            'temperature_mean_threshold': 30,
            'humidity_mean_threshold': 80,
            'traffic_volume_threshold': 1000,
            'value_mean_threshold': 50,
            'value_std_threshold': 40
        }
    }
    
    # Initialize monitor
    monitor = DataMonitor()
    
    # Connect to WebSocket API
    await monitor.connect_api()
    
    # Get absolute paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    
    # Load data files
    data_files = {
        'traffic_light': data_dir / "traffic_raw_siemens_light-veh.json",
        'traffic_heavy': data_dir / "traffic_raw_siemens_heavy-veh.json",
        'environmental': data_dir / "environ_MS83200MS_nowind_3m-10min.json"
    }
    
    try:
        while True:  # Continuous monitoring loop
            for data_type, file_path in data_files.items():
                if not file_path.exists():
                    logging.error(f"Data file not found: {file_path}")
                    continue
                    
                logging.info(f"Processing {data_type} data from {file_path}")
                total_points = 0
                
                # Process file in chunks
                for chunk in load_large_json(str(file_path)):
                    if not chunk:
                        continue
                        
                    total_points += len(chunk)
                    logging.info(f"Processed {total_points} points from {data_type}")
                    
                    # Process the chunk
                    for point in chunk:
                        await process_data_point(monitor, point, data_type)
                    
                    # Small delay between chunks
                    await asyncio.sleep(0.1)
                
                logging.info(f"Completed processing {total_points} points from {data_type}")
            
            # Save metrics after each monitoring cycle
            monitor.save_metrics()
            
            # Wait before starting next monitoring cycle
            await asyncio.sleep(config['sampling_interval'])
            
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user")
    except Exception as e:
        logging.error(f"Error in monitoring loop: {e}")
    finally:
        # Save final metrics
        monitor.save_metrics()
        logging.info("Monitoring stopped")

if __name__ == "__main__":
    asyncio.run(main()) 