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
    monitor = DataMonitor()
    last_update_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            if current_time - last_update_time >= 1:  # Process every second
                try:
                    # Get latest data point
                    if monitor.data_window:
                        latest_data = monitor.data_window[-1]
                        await send_monitoring_data({
                            'statistics': {
                                'timestamp': latest_data['timestamp'].isoformat(),
                                'value': float(latest_data['value'])
                            },
                            'anomalies': [],
                            'data_quality': {
                                'timestamp': latest_data['timestamp'].isoformat(),
                                'completeness': 1.0
                            }
                        })
                except Exception as e:
                    logger.error(f"Error in monitoring: {e}")
                last_update_time = current_time
            await asyncio.sleep(0.1)
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