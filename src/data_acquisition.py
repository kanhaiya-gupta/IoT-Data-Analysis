import pandas as pd
import requests
import os
from pathlib import Path
from typing import Optional, Dict, Any

def fetch_api_data(url):
    """Fetch JSON data from a public API and return a DataFrame."""
    try:
        res = requests.get(url)
        res.raise_for_status()  # Raise exception for bad status codes
        data = res.json()
        return pd.DataFrame(data)
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def load_json_file(file_path):
    """Load a JSON file into a DataFrame."""
    try:
        df = pd.read_json(file_path)
        return df
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None

def load_csv_file(file_path, parse_dates=None):
    """Load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path, parse_dates=parse_dates)
        return df
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return None

def load_iot_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load IoT data from JSON file with proper timestamp handling."""
    try:
        df = pd.read_json(file_path)
        if "timestamp" in df.columns:
            # Convert milliseconds timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"Error loading IoT data from {file_path}: {e}")
        return None

def load_environmental_data() -> Optional[pd.DataFrame]:
    """Load environmental sensor data."""
    env_path = Path("data/environ_MS83200MS_nowind_3m-10min.json")
    df = load_iot_data(str(env_path))
    if df is not None:
        # Rename columns for clarity
        df.columns = [col.lower() for col in df.columns]
    return df

def load_traffic_data() -> Dict[str, Optional[pd.DataFrame]]:
    """Load both light and heavy vehicle traffic data."""
    light_path = Path("data/traffic_raw_siemens_light-veh.json")
    heavy_path = Path("data/traffic_raw_siemens_heavy-veh.json")
    
    df_light = load_iot_data(str(light_path))
    df_heavy = load_iot_data(str(heavy_path))
    
    if df_light is not None:
        df_light.rename(columns={"value": "light_vehicles"}, inplace=True)
    if df_heavy is not None:
        df_heavy.rename(columns={"value": "heavy_vehicles"}, inplace=True)
    
    return {
        "light_vehicles": df_light,
        "heavy_vehicles": df_heavy
    }

def main():
    """Main function to demonstrate IoT data loading."""
    # Load environmental data
    df_env = load_environmental_data()
    if df_env is not None:
        print("\nEnvironmental Data (first 5 rows):")
        print(df_env.head())
        print("\nEnvironmental Data Info:")
        print(df_env.info())
    
    # Load traffic data
    traffic_data = load_traffic_data()
    for vehicle_type, df in traffic_data.items():
        if df is not None:
            print(f"\n{vehicle_type.title()} Data (first 5 rows):")
            print(df.head())
            print(f"\n{vehicle_type.title()} Data Info:")
            print(df.info())

if __name__ == "__main__":
    main()