from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import json
import asyncio
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI(
    title="IoT Monitoring Dashboard",
    description="Real-time monitoring of IoT sensor data",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory for serving HTML/JS files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
js_dir = static_dir / "js"
js_dir.mkdir(exist_ok=True)

# Custom middleware for JavaScript files
class JavaScriptMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.endswith('.js'):
            response.headers["Content-Type"] = "application/javascript; charset=utf-8"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

# Add the JavaScript middleware
app.add_middleware(JavaScriptMiddleware)

# Mount static files with proper configuration
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket clients list
clients = []

async def send_monitoring_data(data: Dict[str, Any]):
    """Send monitoring data to all connected clients."""
    if not clients:
        # Only log this warning once when the server starts
        if not hasattr(send_monitoring_data, '_warned'):
            logger.info("Waiting for dashboard clients to connect at http://127.0.0.1:8001")
            send_monitoring_data._warned = True
        return
        
    # Convert data to JSON
    try:
        message = json.dumps(data, default=str)
        await asyncio.gather(
            *[client.send_text(message) for client in clients]
        )
    except Exception as e:
        logger.error(f"Error sending data: {e}")

@app.get("/")
async def monitoring_dashboard():
    """Serve the monitoring dashboard HTML."""
    try:
        with open("static/index.html", "r") as f:
            html_content = f.read()
    except FileNotFoundError:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>IoT Monitoring Dashboard</title>
        </head>
        <body>
            <h1>Dashboard not found</h1>
            <p>Please ensure the static files are properly set up.</p>
        </body>
        </html>
        """
    
    return HTMLResponse(
        content=html_content,
        headers={
            "Content-Type": "text/html; charset=utf-8",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections."""
    await websocket.accept()
    clients.append(websocket)
    logger.info(f"New WebSocket client connected. Total clients: {len(clients)}")
    
    try:
        while True:
            # Receive and handle messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get('type') == 'ping':
                    logger.debug("Received ping from client")
                    await websocket.send_text(json.dumps({'type': 'pong'}))
                else:
                    logger.debug(f"Received message from client: {message}")
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Remaining clients: {len(clients)}")

async def broadcast_message(message: str):
    """Broadcast a message to all connected WebSocket clients."""
    for client in clients:
        try:
            await client.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            clients.remove(client)

@app.on_event("startup")
async def startup_event():
    """Start the vehicle data processing on startup."""
    logger.info("Starting vehicle data processing...")
    try:
        # Initialize data processing
        asyncio.create_task(process_vehicle_data())
        logger.info("Vehicle data processing started successfully")
    except Exception as e:
        logger.error(f"Failed to start vehicle data processing: {e}")
        raise

# Add debug logging to process_vehicle_data
async def process_vehicle_data():
    """Process vehicle data."""
    try:
        logger.info("Loading vehicle data files...")
        # Read both vehicle data files
        light_data = pd.read_json("data/traffic_raw_siemens_light-veh.json")
        heavy_data = pd.read_json("data/traffic_raw_siemens_heavy-veh.json")
        logger.info(f"Loaded {len(light_data)} light vehicle records and {len(heavy_data)} heavy vehicle records")
        
        # Sort data by timestamp
        light_data = light_data.sort_values('timestamp')
        heavy_data = heavy_data.sort_values('timestamp')
        
        # Process data and send updates
        current_index = 0
        while True:
            try:
                if current_index >= len(light_data) or current_index >= len(heavy_data):
                    current_index = 0  # Reset to start if we reach the end
                
                # Get current data points
                light_current = light_data.iloc[current_index]
                heavy_current = heavy_data.iloc[current_index]
                
                # Prepare statistics data
                stats_data = {
                    'light_vehicle': {
                        'timestamp': light_current['timestamp'],
                        'value': float(light_current['value'])
                    },
                    'heavy_vehicle': {
                        'timestamp': heavy_current['timestamp'],
                        'value': float(heavy_current['value'])
                    }
                }
                
                # Generate some sample anomalies (for demonstration)
                anomalies = []
                if float(light_current['value']) > 100:  # Example threshold
                    anomalies.append({
                        'timestamp': light_current['timestamp'],
                        'type': 'high_traffic',
                        'severity': 'warning',
                        'message': 'High traffic volume detected'
                    })
                
                # Prepare data quality metrics
                data_quality = {
                    'timestamp': light_current['timestamp'],
                    'completeness': 1.0,
                    'accuracy': 0.95,
                    'consistency': 0.98
                }
                
                # Generate some sample alerts (for demonstration)
                alerts = []
                if float(heavy_current['value']) >= 10:  # Changed threshold from 50 to 10
                    alerts.append({
                        'timestamp': heavy_current['timestamp'],
                        'type': 'heavy_vehicle_alert',
                        'severity': 'critical',
                        'message': 'High number of heavy vehicles detected'
                    })
                
                # Send all data via WebSocket
                await send_monitoring_data({
                    'statistics': stats_data,
                    'anomalies': anomalies,
                    'data_quality': data_quality,
                    'alerts': alerts
                })
                
                # Move to next data point
                current_index += 1
                
                # Wait before next update
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing vehicle data: {e}")
                await asyncio.sleep(1)  # Wait before retrying
            
    except Exception as e:
        logger.error(f"Error processing vehicle data: {e}")
        raise 