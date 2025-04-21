import os
import sys
import asyncio
import uvicorn
from monitoring_api import app, send_monitoring_data
from monitoring import DataMonitor
import logging
import signal
import threading
import time
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_fastapi():
    """Run the FastAPI server."""
    try:
        config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="info")
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logger.error(f"Error running FastAPI server: {e}")

async def run_monitoring():
    """Run the monitoring system."""
    config = {
        'window_size': 100,
        'sampling_interval': 1,  # Process every second
        'output_dir': 'outputs/monitoring',
        'alert_thresholds': {
            'completeness_threshold': 0.95,
            'value_mean_threshold': 50,
            'value_std_threshold': 40
        }
    }
    
    monitor = DataMonitor(config)
    last_update_time = time.time()
    
    try:
        # Process data files
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        data_files = {
            'traffic_light': os.path.join(data_dir, "traffic_raw_siemens_light-veh.json"),
            'traffic_heavy': os.path.join(data_dir, "traffic_raw_siemens_heavy-veh.json")
        }
        
        # Load initial data
        initial_data = {}
        for data_type, file_path in data_files.items():
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    initial_data[data_type] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading {data_type} data: {e}")
                continue
        
        data_index = {data_type: 0 for data_type in data_files.keys()}
        
        while True:
            current_time = time.time()
            time_since_last_update = current_time - last_update_time
            
            if time_since_last_update >= config['sampling_interval']:
                # Process one point from each data type
                for data_type, file_path in data_files.items():
                    if data_type not in initial_data:
                        continue
                        
                    data = initial_data[data_type]
                    if data_index[data_type] >= len(data):
                        data_index[data_type] = 0  # Reset to start
                        
                    point = data[data_index[data_type]]
                    data_index[data_type] += 1
                    
                    # Convert timestamp from milliseconds to ISO format
                    timestamp = datetime.fromtimestamp(point['timestamp'] / 1000).isoformat()
                    
                    # Create processed point
                    processed_point = {
                        'timestamp': timestamp,
                        'value': float(point['value']),
                        'type': data_type
                    }
                    
                    # Update data window
                    monitor.update_data_window(processed_point)
                    
                    logger.info(f"Processed {data_type} point: {processed_point}")
                
                # Analyze data if window is full
                if len(monitor.data_window) >= monitor.window_size:
                    metrics = monitor.analyze_data()
                    if metrics:
                        await send_monitoring_data(metrics)
                
                last_update_time = current_time
            
            await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring: {e}")
    finally:
        logger.info("Monitoring stopped")

async def main():
    """Main function to run the monitoring system."""
    try:
        # Start FastAPI server in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi)
        fastapi_thread.daemon = True
        fastapi_thread.start()
        
        # Run monitoring system
        await run_monitoring()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        logger.info("Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main()) 