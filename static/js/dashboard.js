// Set startup flag immediately
window.dashboardScriptStarted = true;

// Test if script is executing
console.log('=== SCRIPT START TEST ===');
console.log('This should be the first line you see');
console.log('=== Dashboard Script Started ===');

// Initialize WebSocket connection
console.log('Attempting to connect to WebSocket server...');
const ws = new WebSocket('ws://localhost:8001/ws');

// Data storage for plots
let statisticsData = [];
let anomaliesData = [];
let qualityData = [];
let alertsData = [];

// Maximum number of points to display
const MAX_POINTS = 100;

// Initialize plots
console.log('Initializing plot elements...');
const statisticsPlot = document.getElementById('statistics-plot');
const anomaliesPlot = document.getElementById('anomalies-plot');
const qualityPlot = document.getElementById('quality-plot');
const alertsPlot = document.getElementById('alerts-plot');

if (!statisticsPlot) {
    console.error('Statistics plot element not found!');
}
if (!anomaliesPlot) {
    console.error('Anomalies plot element not found!');
}
if (!qualityPlot) {
    console.error('Quality plot element not found!');
}
if (!alertsPlot) {
    console.error('Alerts plot element not found!');
}

// Helper function to parse and validate timestamp
function parseTimestamp(timestamp) {
    console.log('Parsing timestamp:', timestamp);
    if (!timestamp) {
        console.warn("No timestamp provided");
        return new Date();
    }
    
    try {
        // If timestamp is already a Date object
        if (timestamp instanceof Date) {
            return timestamp;
        }
        
        // Parse ISO string timestamp
        if (typeof timestamp === 'string') {
            const parsedDate = new Date(timestamp);
            if (!isNaN(parsedDate.getTime())) {
                return parsedDate;
            }
        }
        
        // If timestamp is a number (unix timestamp)
        if (typeof timestamp === 'number') {
            // Convert to milliseconds if needed
            const timestampMs = timestamp < 10000000000 ? timestamp * 1000 : timestamp;
            const parsedDate = new Date(timestampMs);
            if (!isNaN(parsedDate.getTime())) {
                return parsedDate;
            }
        }
        
        console.warn("Invalid timestamp format:", timestamp);
        return new Date();
    } catch (error) {
        console.error("Error parsing timestamp:", error);
        return new Date();
    }
}

// WebSocket event handlers
ws.onopen = () => {
    console.log('=== WebSocket Connection Established ===');
    console.log('Connected to WebSocket server at ws://localhost:8001/ws');
    console.log('WebSocket readyState:', ws.readyState);
};

ws.onmessage = (event) => {
    console.log('=== WebSocket Message ===');
    console.log('Raw message:', event.data);
    
    try {
        const data = JSON.parse(event.data);
        console.log('Parsed data:', data);
        
        // Skip ping/pong messages
        if (data.type === 'pong') {
            return;
        }
        
        // Validate data structure
        if (!data || typeof data !== 'object') {
            console.warn('Invalid data: not an object');
            return;
        }
        
        // Process statistics data
        if (data.statistics) {
            // Process light vehicle data
            if (data.statistics.light_vehicle) {
                statisticsData.push({
                    x: new Date(data.statistics.light_vehicle.timestamp),
                    y: data.statistics.light_vehicle.value,
                    type: 'light_vehicle'
                });
            }
            
            // Process heavy vehicle data
            if (data.statistics.heavy_vehicle) {
                statisticsData.push({
                    x: new Date(data.statistics.heavy_vehicle.timestamp),
                    y: data.statistics.heavy_vehicle.value,
                    type: 'heavy_vehicle'
                });
            }
            
            // Keep only the last MAX_POINTS points
            if (statisticsData.length > MAX_POINTS) {
                statisticsData = statisticsData.slice(-MAX_POINTS);
            }
            
            // Update statistics plot
            const lightVehicleData = statisticsData.filter(d => d.type === 'light_vehicle');
            const heavyVehicleData = statisticsData.filter(d => d.type === 'heavy_vehicle');
            
            Plotly.newPlot(statisticsPlot, [
                {
                    x: lightVehicleData.map(d => d.x),
                    y: lightVehicleData.map(d => d.y),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Light Vehicle',
                    line: {
                        shape: 'spline',
                        smoothing: 0.3
                    }
                },
                {
                    x: heavyVehicleData.map(d => d.x),
                    y: heavyVehicleData.map(d => d.y),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Heavy Vehicle',
                    line: {
                        shape: 'spline',
                        smoothing: 0.3
                    }
                }
            ], {
                title: 'Vehicle Count Over Time',
                xaxis: { 
                    title: 'Time',
                    type: 'date'
                },
                yaxis: { 
                    title: 'Count',
                    range: [0, Math.max(...statisticsData.map(d => d.y)) * 1.1]
                }
            });
        }
        
        // Process anomalies data
        if (data.anomalies && Array.isArray(data.anomalies)) {
            data.anomalies.forEach(anomaly => {
                anomaliesData.push({
                    x: new Date(anomaly.timestamp),
                    y: 1, // Use constant value for visualization
                    type: anomaly.type,
                    severity: anomaly.severity,
                    message: anomaly.message
                });
            });
            
            if (anomaliesData.length > MAX_POINTS) {
                anomaliesData = anomaliesData.slice(-MAX_POINTS);
            }
            
            // Update anomalies plot
            Plotly.newPlot(anomaliesPlot, [{
                x: anomaliesData.map(d => d.x),
                y: anomaliesData.map(d => d.y),
                type: 'scatter',
                mode: 'markers',
                name: 'Anomalies',
                marker: { 
                    color: anomaliesData.map(d => d.severity === 'critical' ? 'red' : 'orange'),
                    size: 10
                },
                text: anomaliesData.map(d => d.message),
                hoverinfo: 'text'
            }], {
                title: 'Detected Anomalies',
                xaxis: { title: 'Time', type: 'date' },
                yaxis: { 
                    title: 'Anomaly',
                    range: [0, 2],
                    showticklabels: false
                }
            });
        }
        
        // Process data quality data
        if (data.data_quality) {
            qualityData.push({
                x: new Date(data.data_quality.timestamp),
                completeness: data.data_quality.completeness,
                accuracy: data.data_quality.accuracy,
                consistency: data.data_quality.consistency
            });
            
            if (qualityData.length > MAX_POINTS) {
                qualityData = qualityData.slice(-MAX_POINTS);
            }
            
            // Update data quality plot
            Plotly.newPlot(qualityPlot, [
                {
                    x: qualityData.map(d => d.x),
                    y: qualityData.map(d => d.completeness),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Completeness'
                },
                {
                    x: qualityData.map(d => d.x),
                    y: qualityData.map(d => d.accuracy),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Accuracy'
                },
                {
                    x: qualityData.map(d => d.x),
                    y: qualityData.map(d => d.consistency),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Consistency'
                }
            ], {
                title: 'Data Quality Metrics',
                xaxis: { title: 'Time', type: 'date' },
                yaxis: { title: 'Score', range: [0, 1] }
            });
        }
        
        // Process alerts data
        if (data.alerts && Array.isArray(data.alerts)) {
            data.alerts.forEach(alert => {
                alertsData.push({
                    x: new Date(alert.timestamp),
                    y: 1, // Use constant value for visualization
                    type: alert.type,
                    severity: alert.severity,
                    message: alert.message
                });
            });
            
            if (alertsData.length > MAX_POINTS) {
                alertsData = alertsData.slice(-MAX_POINTS);
            }
            
            // Update alerts plot
            Plotly.newPlot(alertsPlot, [{
                x: alertsData.map(d => d.x),
                y: alertsData.map(d => d.y),
                type: 'scatter',
                mode: 'markers',
                name: 'Alerts',
                marker: { 
                    color: alertsData.map(d => d.severity === 'critical' ? 'red' : 'orange'),
                    size: 10
                },
                text: alertsData.map(d => d.message),
                hoverinfo: 'text'
            }], {
                title: 'System Alerts',
                xaxis: { title: 'Time', type: 'date' },
                yaxis: { 
                    title: 'Alert',
                    range: [0, 2],
                    showticklabels: false
                }
            });
        }
    } catch (error) {
        console.error('Error in WebSocket message handler:', error);
        console.error('Failed message content:', event.data);
    }
};

ws.onerror = (error) => {
    console.error('=== WebSocket Error ===');
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('=== WebSocket Connection Closed ===');
    console.log('WebSocket connection closed');
};

// Send periodic ping to keep connection alive
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000); 