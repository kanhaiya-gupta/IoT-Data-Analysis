import paho.mqtt.subscribe as subscribe
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/mqtt_stream.log"),
        logging.StreamHandler()
    ]
)

# Configuration
MQTT_HOST = "mqtt.datacamp.com"
MQTT_TOPIC = "datacamp/iot/simple"
MAX_CACHE = 1000
CACHE_TIMEOUT = 300  # seconds

class IoTDataStream:
    def __init__(self, host: str, topic: str, max_cache: int = 1000):
        self.host = host
        self.topic = topic
        self.max_cache = max_cache
        self.cache = []
        self.last_save_time = time.time()
        self.output_dir = Path("../outputs/streaming")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_message(self, message: Any) -> Optional[Dict[str, Any]]:
        """Process and validate MQTT message."""
        try:
            # Parse JSON payload
            payload = json.loads(message.payload.decode('utf-8'))
            
            # Add timestamp
            payload['mqtt_timestamp'] = datetime.fromtimestamp(message.timestamp)
            
            # Validate required fields
            required_fields = ['timestamp', 'value']
            if not all(field in payload for field in required_fields):
                logging.warning(f"Missing required fields in message: {payload}")
                return None
            
            return payload
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON payload: {message.payload}")
            return None
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            return None
    
    def save_cache(self):
        """Save cached data to file."""
        if not self.cache:
            return
        
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"datastream_{timestamp}.csv"
            
            # Convert cache to DataFrame
            df = pd.DataFrame(self.cache)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            logging.info(f"Saved {len(self.cache)} messages to {output_file}")
            
            # Clear cache
            self.cache.clear()
            self.last_save_time = time.time()
        except Exception as e:
            logging.error(f"Error saving cache: {e}")
    
    def on_message(self, client, userdata, message):
        """Callback function to process MQTT messages."""
        try:
            # Process message
            data = self.process_message(message)
            if data is None:
                return
            
            # Add to cache
            self.cache.append(data)
            
            # Check cache size or timeout
            current_time = time.time()
            if (len(self.cache) >= self.max_cache or 
                current_time - self.last_save_time >= CACHE_TIMEOUT):
                self.save_cache()
        except Exception as e:
            logging.error(f"Error in on_message callback: {e}")
    
    def start_streaming(self):
        """Start MQTT subscription."""
        try:
            logging.info(f"Starting MQTT subscription to {self.topic} on {self.host}")
            subscribe.callback(
                self.on_message,
                topics=self.topic,
                hostname=self.host
            )
        except Exception as e:
            logging.error(f"Error starting MQTT subscription: {e}")

def main():
    """Main function to start IoT data streaming."""
    try:
        # Initialize stream
        stream = IoTDataStream(MQTT_HOST, MQTT_TOPIC, MAX_CACHE)
        
        # Start streaming
        stream.start_streaming()
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()