import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/visualization.log'),
        logging.StreamHandler()
    ]
)

class DataVisualizer:
    def __init__(self, output_dir: str = "outputs/streaming"):
        """Initialize the visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Set up consistent plot style."""
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (15, 8)
        plt.rcParams['font.size'] = 12
    
    def load_data(self, data: List[Dict[str, Any]], sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load and preprocess data with optional sampling for large datasets."""
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        if sample_size and len(df) > sample_size:
            # Systematic sampling to maintain temporal patterns
            step = len(df) // sample_size
            df = df.iloc[::step]
            print(f"Sampled data from {len(data)} to {len(df)} points")
        
        return df

    def resample_time_series(self, df: pd.DataFrame, rule: str = '1H') -> pd.DataFrame:
        """Resample time series data to reduce size and smooth patterns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        resampled = df[numeric_cols].resample(rule).agg({
            col: 'mean' for col in numeric_cols
        }).ffill()
        print(f"Resampled data from {len(df)} to {len(resampled)} points using {rule} intervals")
        return resampled

    def plot_environmental_time_series(self, df: pd.DataFrame):
        """Plot time series of environmental variables with efficient data handling."""
        # Resample data for smoother visualization
        df_resampled = self.resample_time_series(df, '1H')
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Environmental Variables Time Series', fontsize=16)
        
        variables = {
            (0, 0): ('temperature', 'Temperature (°C)', 'red'),
            (0, 1): ('humidity', 'Humidity (%)', 'blue'),
            (1, 0): ('radiation', 'Radiation', 'orange'),
            (1, 1): ('pressure', 'Pressure', 'green'),
            (2, 0): ('sunshine', 'Sunshine', 'yellow'),
            (2, 1): ('precipitation', 'Precipitation', 'purple')
        }
        
        for (i, j), (var, label, color) in variables.items():
            if var in df_resampled.columns:
                df_resampled[var].plot(ax=axes[i, j], color=color)
                axes[i, j].set_title(label)
                axes[i, j].set_xlabel('Time')
                axes[i, j].set_ylabel(label)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'environmental_time_series.png', dpi=300)
        plt.close()
    
    def plot_traffic_time_series(self, df: pd.DataFrame):
        """Plot detailed time series of traffic variables."""
        try:
            # Create subplots for different time aggregations
            fig, axes = plt.subplots(3, 1, figsize=(15, 15))
            fig.suptitle('Traffic Time Series Analysis', fontsize=16)
            
            # Hourly traffic
            hourly = df.resample('H').mean()
            hourly['light_vehicles'].plot(ax=axes[0], color='blue', label='Light Vehicles')
            hourly['heavy_vehicles'].plot(ax=axes[0], color='red', label='Heavy Vehicles')
            axes[0].set_title('Hourly Traffic Patterns')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Vehicle Count')
            axes[0].legend()
            
            # Daily traffic
            daily = df.resample('D').mean()
            daily['light_vehicles'].plot(ax=axes[1], color='blue', label='Light Vehicles')
            daily['heavy_vehicles'].plot(ax=axes[1], color='red', label='Heavy Vehicles')
            axes[1].set_title('Daily Traffic Patterns')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Average Vehicle Count')
            axes[1].legend()
            
            # Weekly traffic
            weekly = df.resample('W').mean()
            weekly['light_vehicles'].plot(ax=axes[2], color='blue', label='Light Vehicles')
            weekly['heavy_vehicles'].plot(ax=axes[2], color='red', label='Heavy Vehicles')
            axes[2].set_title('Weekly Traffic Patterns')
            axes[2].set_xlabel('Week')
            axes[2].set_ylabel('Average Vehicle Count')
            axes[2].legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'traffic_time_series.png', dpi=300)
            plt.close()
            logging.info("Generated traffic time series plots")
        except Exception as e:
            logging.error(f"Error plotting traffic time series: {e}")

    def plot_traffic_distributions(self, df: pd.DataFrame):
        """Plot statistical distributions of traffic data."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Traffic Distribution Analysis', fontsize=16)
            
            # Light vehicles distribution
            sns.histplot(data=df, x='light_vehicles', ax=axes[0, 0], kde=True)
            axes[0, 0].set_title('Light Vehicles Distribution')
            axes[0, 0].set_xlabel('Vehicle Count')
            
            # Heavy vehicles distribution
            sns.histplot(data=df, x='heavy_vehicles', ax=axes[0, 1], kde=True)
            axes[0, 1].set_title('Heavy Vehicles Distribution')
            axes[0, 1].set_xlabel('Vehicle Count')
            
            # Box plots
            sns.boxplot(data=df[['light_vehicles', 'heavy_vehicles']], ax=axes[1, 0])
            axes[1, 0].set_title('Vehicle Count Box Plots')
            axes[1, 0].set_ylabel('Vehicle Count')
            
            # Violin plots
            sns.violinplot(data=df[['light_vehicles', 'heavy_vehicles']], ax=axes[1, 1])
            axes[1, 1].set_title('Vehicle Count Violin Plots')
            axes[1, 1].set_ylabel('Vehicle Count')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'traffic_distributions.png', dpi=300)
            plt.close()
            logging.info("Generated traffic distribution plots")
        except Exception as e:
            logging.error(f"Error plotting traffic distributions: {e}")

    def plot_seasonal_decomposition(self, df: pd.DataFrame, column: str, period: int = 24):
        """Perform and plot seasonal decomposition of time series data."""
        try:
            plt.figure(figsize=(12, 10))
            # Fill missing values for decomposition
            series = df[column].ffill()
            
            # Check if we have enough data points for decomposition
            if len(series) >= 2 * period:
                decomp = sm.tsa.seasonal_decompose(series, period=period)
                
                plt.subplot(411)
                plt.plot(series, label='Original')
                plt.legend()
                plt.title(f'Seasonal Decomposition of {column}')
                
                plt.subplot(412)
                plt.plot(decomp.trend, label='Trend')
                plt.legend()
                
                plt.subplot(413)
                plt.plot(decomp.seasonal, label='Seasonal')
                plt.legend()
                
                plt.subplot(414)
                plt.plot(decomp.resid, label='Residual')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'seasonal_decomposition_{column}.png', dpi=300)
                plt.close()
                logging.info(f"Generated seasonal decomposition plot for {column}")
            else:
                logging.warning(f"Not enough data points for seasonal decomposition of {column}")
        except Exception as e:
            logging.error(f"Error in seasonal decomposition: {e}")

    def plot_missing_data(self, df: pd.DataFrame):
        """Plot missing data patterns with enhanced visualization."""
        try:
            # Calculate missing data statistics
            missing_stats = df.isnull().sum()
            missing_percent = (missing_stats / len(df)) * 100
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('Missing Data Analysis', fontsize=16)
            
            # Heatmap of missing data
            sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax1)
            ax1.set_title('Missing Data Pattern')
            ax1.set_xlabel('Variables')
            ax1.set_ylabel('Time Points')
            
            # Bar plot of missing percentages
            missing_percent.plot(kind='bar', ax=ax2, color='red')
            ax2.set_title('Percentage of Missing Values by Variable')
            ax2.set_xlabel('Variables')
            ax2.set_ylabel('Percentage Missing')
            ax2.set_ylim(0, 100)
            
            # Add percentage labels on bars
            for i, v in enumerate(missing_percent):
                ax2.text(i, v + 1, f'{v:.1f}%', ha='center')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'missing_data_pattern.png', dpi=300)
            plt.close()
            logging.info("Generated missing data analysis plots")
        except Exception as e:
            logging.error(f"Error plotting missing data patterns: {e}")

    def plot_correlations(self, df: pd.DataFrame, title: str = 'Correlation Matrix'):
        """Plot correlation heatmap."""
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()
    
    def plot_distributions(self, df: pd.DataFrame, title: str = 'Variable Distributions'):
        """Plot distributions of variables."""
        n_cols = 2
        n_rows = (len(df.columns) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle(title, fontsize=16)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']
        
        for idx, (col, color) in enumerate(zip(df.columns, colors)):
            if n_rows == 1:
                ax = axes[idx % n_cols]
            else:
                row = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row, col_idx]
            
            sns.histplot(df[col], ax=ax, color=color, kde=True)
            ax.set_title(f'{col} Distribution')
        
        # Remove empty subplots
        for idx in range(len(df.columns), n_rows * n_cols):
            if n_rows == 1:
                fig.delaxes(axes[idx % n_cols])
            else:
                row = idx // n_cols
                col_idx = idx % n_cols
                fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()
    
    def plot_relationships(self, df: pd.DataFrame, title: str = 'Variable Relationships'):
        """Plot relationships between variables."""
        n_vars = len(df.columns)
        n_plots = min(4, (n_vars * (n_vars - 1)) // 2)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        plot_idx = 0
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if plot_idx >= n_plots:
                    break
                row = plot_idx // 2
                col = plot_idx % 2
                sns.scatterplot(data=df, x=df.columns[i], y=df.columns[j], ax=axes[row, col])
                axes[row, col].set_title(f'{df.columns[i]} vs {df.columns[j]}')
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{title.lower().replace(" ", "_")}.png')
        plt.close()
    
    def plot_environmental_traffic_correlations(self, env_df: pd.DataFrame, traffic_df: pd.DataFrame):
        """Plot correlations between environmental factors and traffic with efficient data handling."""
        try:
            # Resample both dataframes to hourly intervals for correlation analysis
            env_resampled = self.resample_time_series(env_df, '1H')
            traffic_resampled = self.resample_time_series(traffic_df, '1H')
            
            # Merge resampled dataframes
            df = pd.merge(env_resampled, traffic_resampled, left_index=True, right_index=True)
            
            # Sample data if too large
            if len(df) > 1000:
                df = df.sample(1000)
                print("Sampled correlation data to 1000 points")
            
            env_vars = ['temperature', 'humidity', 'radiation', 'pressure']
            traffic_vars = ['light_vehicles', 'heavy_vehicles']
            
            fig, axes = plt.subplots(len(env_vars), len(traffic_vars), figsize=(15, 20))
            fig.suptitle('Environmental Factors vs Traffic', fontsize=16)
            
            for i, env_var in enumerate(env_vars):
                for j, traffic_var in enumerate(traffic_vars):
                    if env_var in df.columns and traffic_var in df.columns:
                        clean_data = df[[env_var, traffic_var]].dropna()
                        sns.scatterplot(data=clean_data, x=env_var, y=traffic_var, ax=axes[i, j])
                        axes[i, j].set_title(f'{env_var} vs {traffic_var}')
                        
                        if len(clean_data) > 1:
                            z = np.polyfit(clean_data[env_var], clean_data[traffic_var], 1)
                            p = np.poly1d(z)
                            axes[i, j].plot(clean_data[env_var], p(clean_data[env_var]), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'environmental_traffic_correlations.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not plot environmental traffic correlations: {str(e)}")

    def visualize_all(self, environmental_data: List[Dict[str, Any]], traffic_data: List[Dict[str, Any]]):
        """Generate all visualizations."""
        try:
            # Load and preprocess data
            env_df = self.load_data(environmental_data)
            traffic_df = self.load_data(traffic_data)
            
            # Generate all visualizations
            self.plot_environmental_time_series(env_df)
            self.plot_traffic_time_series(traffic_df)
            self.plot_traffic_distributions(traffic_df)
            self.plot_correlations(env_df, 'Environmental Correlations')
            self.plot_environmental_traffic_correlations(env_df, traffic_df)
            
            # Seasonal decomposition for key variables
            for col in ['temperature', 'humidity', 'light_vehicles', 'heavy_vehicles']:
                if col in env_df.columns or col in traffic_df.columns:
                    df = env_df if col in env_df.columns else traffic_df
                    self.plot_seasonal_decomposition(df, col)
            
            # Missing data analysis
            self.plot_missing_data(pd.concat([env_df, traffic_df], axis=1))
            
            logging.info("All visualizations generated successfully")
        except Exception as e:
            logging.error(f"Error in visualize_all: {e}")

def load_json_data(file_path):
    """Load JSON data from file with validation."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # Log data statistics
        logging.info(f"Loaded {file_path}")
        logging.info(f"Number of records: {len(df)}")
        logging.info(f"Columns: {df.columns.tolist()}")
        logging.info(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Data validation
        if 'timestamp' in df.columns:
            logging.info(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            # Convert timestamp to datetime for better analysis
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            logging.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            logging.info(f"Total days: {(df['datetime'].max() - df['datetime'].min()).days}")
        
        if 'value' in df.columns:
            logging.info(f"Value statistics:\n{df['value'].describe()}")
            logging.info(f"Number of unique values: {df['value'].nunique()}")
            logging.info(f"Number of zero values: {(df['value'] == 0).sum()}")
        
        return df
    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        return None

def setup_plot_style():
    """Set up the plotting style."""
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['font.size'] = 12

def plot_traffic_patterns(light_df, heavy_df):
    """Plot detailed traffic patterns with data validation."""
    logging.info("\nProcessing traffic patterns:")
    logging.info(f"Light vehicles data points: {len(light_df)}")
    logging.info(f"Heavy vehicles data points: {len(heavy_df)}")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(4, 2)
    
    # 1. Hourly patterns
    ax1 = fig.add_subplot(gs[0, :])
    light_hourly = light_df.groupby(light_df['datetime'].dt.hour)['value'].agg(['mean', 'std', 'count'])
    heavy_hourly = heavy_df.groupby(heavy_df['datetime'].dt.hour)['value'].agg(['mean', 'std', 'count'])
    
    ax1.errorbar(light_hourly.index, light_hourly['mean'], yerr=light_hourly['std'], 
                fmt='b-', label='Light Vehicles', capsize=5)
    ax1.errorbar(heavy_hourly.index, heavy_hourly['mean'], yerr=heavy_hourly['std'], 
                fmt='r-', label='Heavy Vehicles', capsize=5)
    ax1.set_title('Average Traffic by Hour with Standard Deviation')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Number of Vehicles')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Daily patterns
    ax2 = fig.add_subplot(gs[1, :])
    light_daily = light_df.groupby(light_df['datetime'].dt.date)['value'].agg(['mean', 'std', 'count'])
    heavy_daily = heavy_df.groupby(heavy_df['datetime'].dt.date)['value'].agg(['mean', 'std', 'count'])
    
    ax2.errorbar(range(len(light_daily)), light_daily['mean'], yerr=light_daily['std'], 
                fmt='b-', label='Light Vehicles', capsize=5)
    ax2.errorbar(range(len(heavy_daily)), heavy_daily['mean'], yerr=heavy_daily['std'], 
                fmt='r-', label='Heavy Vehicles', capsize=5)
    ax2.set_title('Daily Traffic Patterns with Standard Deviation')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Number of Vehicles')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Weekly patterns
    ax3 = fig.add_subplot(gs[2, :])
    light_weekly = light_df.groupby(light_df['datetime'].dt.dayofweek)['value'].agg(['mean', 'std', 'count'])
    heavy_weekly = heavy_df.groupby(heavy_df['datetime'].dt.dayofweek)['value'].agg(['mean', 'std', 'count'])
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ax3.errorbar(days, light_weekly['mean'], yerr=light_weekly['std'], 
                fmt='b-o', label='Light Vehicles', capsize=5)
    ax3.errorbar(days, heavy_weekly['mean'], yerr=heavy_weekly['std'], 
                fmt='r-o', label='Heavy Vehicles', capsize=5)
    ax3.set_title('Weekly Traffic Patterns with Standard Deviation')
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Number of Vehicles')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Distribution plots
    ax4 = fig.add_subplot(gs[3, 0])
    sns.histplot(data=light_df, x='value', bins=50, color='blue', alpha=0.5, label='Light Vehicles', ax=ax4)
    sns.histplot(data=heavy_df, x='value', bins=50, color='red', alpha=0.5, label='Heavy Vehicles', ax=ax4)
    ax4.set_title('Distribution of Vehicle Counts')
    ax4.set_xlabel('Number of Vehicles')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    # 5. Box plots
    ax5 = fig.add_subplot(gs[3, 1])
    # Create a DataFrame for boxplot
    boxplot_data = pd.DataFrame({
        'Light Vehicles': light_df['value'],
        'Heavy Vehicles': heavy_df['value']
    })
    sns.boxplot(data=boxplot_data, ax=ax5)
    ax5.set_title('Box Plot of Vehicle Counts')
    ax5.set_ylabel('Number of Vehicles')
    
    plt.tight_layout()
    plt.savefig('outputs/streaming/traffic_patterns.png', dpi=300)
    plt.close()

def plot_environmental_correlations(env_df):
    """Plot correlations between environmental variables with validation."""
    logging.info("\nProcessing environmental correlations:")
    logging.info(f"Environmental data points: {len(env_df)}")
    
    # Select numerical columns
    numerical_cols = env_df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'timestamp']
    logging.info(f"Numerical columns: {numerical_cols}")
    
    # Calculate correlation matrix
    corr_matrix = env_df[numerical_cols].corr()
    logging.info("\nCorrelation matrix:")
    logging.info(f"\n{corr_matrix}")
    
    # Create heatmap with annotations
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                square=True, linewidths=0.5)
    plt.title('Environmental Variables Correlation Matrix')
    plt.tight_layout()
    plt.savefig('outputs/streaming/environmental_correlations.png', dpi=300)
    plt.close()

def plot_environmental_time_series(env_df):
    """Plot time series of environmental variables with validation."""
    logging.info("\nProcessing environmental time series:")
    
    # Create subplots for each environmental variable
    variables = [col for col in env_df.columns if col not in ['timestamp', 'datetime']]
    fig, axes = plt.subplots(len(variables), 1, figsize=(15, 4*len(variables)))
    
    for i, var in enumerate(variables):
        # Calculate daily statistics
        daily_stats = env_df.groupby(env_df['datetime'].dt.date)[var].agg(['mean', 'std', 'count'])
        
        # Plot with error bars
        axes[i].errorbar(range(len(daily_stats)), daily_stats['mean'], 
                        yerr=daily_stats['std'], fmt='-o', capsize=5)
        axes[i].set_title(f'{var} - Daily Average with Standard Deviation')
        axes[i].set_xlabel('Day')
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
        
        # Log statistics
        logging.info(f"\n{var} statistics:")
        logging.info(f"Number of daily averages: {len(daily_stats)}")
        logging.info(f"Value range: {daily_stats['mean'].min():.2f} to {daily_stats['mean'].max():.2f}")
        logging.info(f"Average standard deviation: {daily_stats['std'].mean():.2f}")
    
    plt.tight_layout()
    plt.savefig('outputs/streaming/environmental_time_series.png', dpi=300)
    plt.close()

def plot_traffic_vs_environment(light_df, heavy_df, env_df):
    """Plot relationships between traffic and environmental conditions with validation."""
    logging.info("\nProcessing traffic vs environment relationships:")
    
    # Merge datasets on nearest timestamp
    merged_light = pd.merge_asof(light_df.sort_values('datetime'), 
                                env_df.sort_values('datetime'),
                                on='datetime',
                                direction='nearest')
    
    merged_heavy = pd.merge_asof(heavy_df.sort_values('datetime'),
                                env_df.sort_values('datetime'),
                                on='datetime',
                                direction='nearest')
    
    logging.info(f"Merged data points - Light vehicles: {len(merged_light)}")
    logging.info(f"Merged data points - Heavy vehicles: {len(merged_heavy)}")
    
    # Create scatter plots with regression lines
    env_vars = ['temperature', 'humidity', 'precipitation', 'pressure']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, var in enumerate(env_vars):
        if var in merged_light.columns:
            # Plot scatter points
            sns.scatterplot(data=merged_light, x=var, y='value', alpha=0.5, 
                           label='Light Vehicles', ax=axes[i])
            sns.scatterplot(data=merged_heavy, x=var, y='value', alpha=0.5,
                           label='Heavy Vehicles', ax=axes[i])
            
            # Add regression lines
            sns.regplot(data=merged_light, x=var, y='value', scatter=False, 
                       color='blue', ax=axes[i])
            sns.regplot(data=merged_heavy, x=var, y='value', scatter=False, 
                       color='red', ax=axes[i])
            
            # Calculate and log correlation coefficients
            light_corr = merged_light[[var, 'value']].corr().iloc[0, 1]
            heavy_corr = merged_heavy[[var, 'value']].corr().iloc[0, 1]
            
            logging.info(f"\nCorrelation with {var}:")
            logging.info(f"Light vehicles: {light_corr:.3f}")
            logging.info(f"Heavy vehicles: {heavy_corr:.3f}")
            
            axes[i].set_title(f'Traffic vs {var.capitalize()}\nCorr: Light={light_corr:.2f}, Heavy={heavy_corr:.2f}')
            axes[i].set_xlabel(var.capitalize())
            axes[i].set_ylabel('Number of Vehicles')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/streaming/traffic_vs_environment.png', dpi=300)
    plt.close()

def main():
    logging.info("Starting data visualization...")
    
    # Set up plotting style
    setup_plot_style()
    
    # Load data
    logging.info("Loading data files...")
    light_df = load_json_data('data/traffic_raw_siemens_light-veh.json')
    heavy_df = load_json_data('data/traffic_raw_siemens_heavy-veh.json')
    env_df = load_json_data('data/environ_MS83200MS_nowind_3m-10min.json')
    
    if all([light_df is not None, heavy_df is not None, env_df is not None]):
        logging.info("Generating traffic pattern visualizations...")
        plot_traffic_patterns(light_df, heavy_df)
        
        logging.info("Generating environmental correlation visualizations...")
        plot_environmental_correlations(env_df)
        
        logging.info("Generating environmental time series visualizations...")
        plot_environmental_time_series(env_df)
        
        logging.info("Generating traffic vs environment visualizations...")
        plot_traffic_vs_environment(light_df, heavy_df, env_df)
        
        logging.info("Visualization complete. Check outputs/streaming directory for results.")
    else:
        logging.error("Failed to load one or more data files.")

if __name__ == "__main__":
    main() 