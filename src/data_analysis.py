import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_analysis.log'),
        logging.StreamHandler()
    ]
)

def load_json_data(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        return None

def analyze_traffic_data(light_veh_data, heavy_veh_data):
    """Analyze traffic data patterns."""
    # Convert to DataFrames
    light_df = pd.DataFrame(light_veh_data)
    heavy_df = pd.DataFrame(heavy_veh_data)
    
    # Basic statistics
    logging.info("\nLight Vehicle Statistics:")
    logging.info(light_df.describe())
    
    logging.info("\nHeavy Vehicle Statistics:")
    logging.info(heavy_df.describe())
    
    # Plot traffic patterns
    plt.figure(figsize=(12, 6))
    plt.plot(light_df.index, light_df['value'], label='Light Vehicles')
    plt.plot(heavy_df.index, heavy_df['value'], label='Heavy Vehicles')
    plt.title('Traffic Patterns Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Vehicles')
    plt.legend()
    plt.savefig('outputs/streaming/traffic_patterns.png')
    plt.close()

def analyze_environmental_data(env_data):
    """Analyze environmental data patterns."""
    env_df = pd.DataFrame(env_data)
    
    # Basic statistics
    logging.info("\nEnvironmental Data Statistics:")
    logging.info(env_df.describe())
    
    # Plot environmental variables
    plt.figure(figsize=(12, 8))
    for column in env_df.columns:
        if column != 'timestamp':
            plt.plot(env_df['timestamp'], env_df[column], label=column)
    plt.title('Environmental Variables Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('outputs/streaming/environmental_patterns.png')
    plt.close()

def main():
    # Load data
    logging.info("Loading data files...")
    light_veh_data = load_json_data('data/traffic_raw_siemens_light-veh.json')
    heavy_veh_data = load_json_data('data/traffic_raw_siemens_heavy-veh.json')
    env_data = load_json_data('data/environ_MS83200MS_nowind_3m-10min.json')
    
    if all([light_veh_data, heavy_veh_data, env_data]):
        # Analyze traffic data
        logging.info("Analyzing traffic data...")
        analyze_traffic_data(light_veh_data, heavy_veh_data)
        
        # Analyze environmental data
        logging.info("Analyzing environmental data...")
        analyze_environmental_data(env_data)
        
        logging.info("Analysis complete. Check the outputs/streaming directory for visualizations.")
    else:
        logging.error("Failed to load one or more data files.")

if __name__ == "__main__":
    main() 