# IoT Data Analysis for Industry 4.0

This project focuses on analyzing IoT data from environmental and traffic sensors to predict traffic patterns using machine learning. The system combines real-time data streaming with historical data analysis to provide insights into traffic behavior based on environmental conditions.

## Project Structure

```
IoT-Data-Analysis/
├── data/                      # Data files
│   ├── environ_MS83200MS_nowind_3m-10min.json
│   ├── traffic_raw_siemens_light-veh.json
│   └── traffic_raw_siemens_heavy-veh.json
├── logs/                      # Log files
├── models/                    # Trained models
├── outputs/                   # Generated outputs
│   └── streaming/            # Real-time analysis results
├── src/                      # Source code
│   ├── data_acquisition.py   # Data loading and preprocessing
│   ├── data_processing.py    # Data processing utilities
│   ├── data_analysis.py      # Data analysis and visualization
│   ├── model_training.py     # Model training and evaluation
│   └── mqtt_streaming.py     # Real-time data streaming
└── requirements.txt          # Project dependencies
```

## Data Sources

### Environmental Data
- **Source**: MS83200MS sensor
- **Format**: JSON
- **Variables**:
  - Temperature (°C)
  - Humidity (%)
  - Radiation (W/m²)
  - Pressure (hPa)
  - Sunshine (minutes)
  - Precipitation (mm)

### Traffic Data
- **Source**: Siemens sensors
- **Format**: JSON
- **Categories**:
  - Light vehicles
  - Heavy vehicles
- **Metrics**:
  - Vehicle count per 10-minute interval
  - Timestamp-based measurements

## Features

### Data Processing
- Time-based feature engineering
- Environmental interaction features
- Rolling window statistics
- Data cleaning and normalization

### Model Training
- Random Forest Classifier
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Feature importance analysis

### Visualizations
1. **Traffic Patterns**
   - Time series plots of vehicle counts
   - Distribution of traffic by vehicle type
   - Daily and weekly traffic patterns

2. **Environmental Analysis**
   - Temperature and humidity trends
   - Radiation and pressure correlations
   - Environmental variable distributions

3. **Model Evaluation**
   - ROC curves
   - Feature importance plots
   - Confusion matrices

## IoT Data Visualization

### Real-time Data Streams
The system processes and visualizes IoT data in real-time, providing insights into:
- Traffic flow patterns
- Environmental conditions
- Correlation between variables

Example visualizations from the streaming outputs:

1. **Traffic Flow Analysis**
   ```
   outputs/streaming/traffic_patterns.png
   ```
   ![Traffic Patterns](outputs/streaming/traffic_patterns.png)
   - Shows hourly distribution of vehicles
   - Compares light vs. heavy vehicle patterns
   - Identifies peak traffic hours

2. **Environmental Conditions**
   ```
   outputs/streaming/environmental_correlations.png
   ```
   ![Environmental Correlations](outputs/streaming/environmental_correlations.png)
   - Temperature and humidity trends
   - Radiation levels throughout the day
   - Pressure variations

3. **Traffic-Environment Correlation**
   ```
   outputs/streaming/traffic_environment_correlation.png
   ```
   ![Traffic-Environment Correlation](outputs/streaming/traffic_environment_correlation.png)
   - Relationship between weather and traffic
   - Impact of environmental factors on vehicle flow
   - Seasonal patterns in traffic behavior

### Understanding IoT Data
IoT (Internet of Things) data in this project represents:
1. **Sensor Measurements**
   - Real-time environmental readings
   - Continuous traffic monitoring
   - Time-stamped data points

2. **Data Patterns**
   - Daily cycles in traffic
   - Weather-related variations
   - Seasonal trends

3. **Predictive Insights**
   - Traffic prediction based on weather
   - Pattern recognition
   - Anomaly detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/IoT-Data-Analysis.git
cd IoT-Data-Analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Analysis
```bash
python src/data_analysis.py
```
This will generate visualizations and statistical analysis of the data.

### Model Training
```bash
python src/model_training.py
```
This will:
- Load and preprocess the data
- Train the model
- Generate evaluation metrics
- Save the trained model

### Real-time Analysis
```bash
python src/mqtt_streaming.py
```
This will:
- Connect to MQTT broker
- Stream real-time data
- Make predictions
- Update visualizations

## Outputs

The system generates several types of outputs:

1. **Model Files**
   - Trained model (`.pkl`)
   - Model metadata (`.json`)
   - Feature importance plots
   - ROC curves

2. **Analysis Results**
   - Traffic pattern visualizations
   - Environmental correlation plots
   - Statistical summaries

3. **Logs**
   - Training progress
   - Model performance metrics
   - Error tracking

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- paho-mqtt
- pathlib
- logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Siemens for traffic data
- MS83200MS sensor for environmental data
- Open-source community for libraries and tools
