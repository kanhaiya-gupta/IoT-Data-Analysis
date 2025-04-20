import pandas as pd
import os
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path

def clean_data(df, method="drop"):
    """Clean DataFrame by handling missing values."""
    if method == "drop":
        return df.dropna()
    elif method == "ffill":
        return df.fillna(method="ffill")
    else:
        raise ValueError("Method must be 'drop' or 'ffill'")

def resample_data(df, interval="10min", agg_method="last"):
    """Resample time-series data to a specified interval."""
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime")
    return df.resample(interval).agg(agg_method)

def pivot_data(df, index="ts", columns="device", values="val"):
    """Pivot DataFrame to restructure by device and timestamp."""
    return pd.pivot_table(df, index=index, columns=columns, values=values)

def concatenate_dataframes(df_list, axis=1):
    """Concatenate a list of DataFrames."""
    return pd.concat(df_list, axis=axis)

def save_dataframe(df, file_path, file_format="csv"):
    """Save DataFrame to CSV or JSON."""
    try:
        if file_format == "csv":
            df.to_csv(file_path, index=False)
        elif file_format == "json":
            df.to_json(file_path, orient="records")
        else:
            raise ValueError("File format must be 'csv' or 'json'")
        print(f"Saved DataFrame to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to {file_path}: {e}")

def clean_iot_data(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Clean IoT data by handling missing values and outliers."""
    # Handle missing values
    if method == "drop":
        df_clean = df.dropna()
    elif method == "ffill":
        df_clean = df.ffill()
    else:
        raise ValueError("Method must be 'drop' or 'ffill'")
    
    # Remove outliers using IQR method
    for column in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[column]):
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
    
    return df_clean

def resample_iot_data(df: pd.DataFrame, interval: str = "10min", agg_method: str = "mean") -> pd.DataFrame:
    """Resample IoT time-series data to a specified interval."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be datetime")
    
    # Define aggregation methods for different columns
    agg_dict = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if agg_method == "mean":
                agg_dict[column] = "mean"
            elif agg_method == "sum":
                agg_dict[column] = "sum"
            elif agg_method == "max":
                agg_dict[column] = "max"
            else:
                agg_dict[column] = "last"
    
    return df.resample(interval).agg(agg_dict)

def combine_iot_data(env_df: pd.DataFrame, traffic_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine environmental and traffic data."""
    # Create a list of all DataFrames
    df_list = [env_df]
    for df in traffic_data.values():
        if df is not None:
            df_list.append(df)
    
    # Combine all DataFrames
    combined_df = pd.concat(df_list, axis=1)
    
    # Forward fill missing values
    combined_df = combined_df.ffill()
    
    return combined_df

def calculate_iot_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculate various IoT metrics and statistics."""
    metrics = {}
    
    # Basic statistics
    metrics["basic_stats"] = df.describe()
    
    # Hourly averages
    metrics["hourly_avg"] = df.resample("h").mean()
    
    # Daily maximums
    metrics["daily_max"] = df.resample("D").max()
    
    # Correlation matrix
    metrics["correlation"] = df.corr()
    
    return metrics

def save_iot_data(df: pd.DataFrame, file_path: str, file_format: str = "csv"):
    """Save IoT data to file."""
    try:
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_format == "csv":
            df.to_csv(output_path, index=True)
        elif file_format == "json":
            df.to_json(output_path, orient="records")
        else:
            raise ValueError("File format must be 'csv' or 'json'")
        print(f"Saved IoT data to {file_path}")
    except Exception as e:
        print(f"Error saving IoT data to {file_path}: {e}")

def main():
    """Main function to demonstrate IoT data processing."""
    from data_acquisition import load_environmental_data, load_traffic_data
    
    # Load data
    df_env = load_environmental_data()
    traffic_data = load_traffic_data()
    
    if df_env is not None and all(df is not None for df in traffic_data.values()):
        # Clean data
        df_env_clean = clean_iot_data(df_env)
        traffic_clean = {k: clean_iot_data(v) for k, v in traffic_data.items()}
        
        # Resample data
        df_env_resampled = resample_iot_data(df_env_clean)
        traffic_resampled = {k: resample_iot_data(v) for k, v in traffic_clean.items()}
        
        # Combine data
        combined_df = combine_iot_data(df_env_resampled, traffic_resampled)
        
        # Calculate metrics
        metrics = calculate_iot_metrics(combined_df)
        
        # Save processed data
        save_iot_data(combined_df, "outputs/streaming/processed_iot_data.csv")
        
        # Print some metrics
        print("\nBasic Statistics:")
        print(metrics["basic_stats"])
        print("\nCorrelation Matrix:")
        print(metrics["correlation"])

if __name__ == "__main__":
    main()