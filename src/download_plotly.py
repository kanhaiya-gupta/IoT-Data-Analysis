import requests
import os
from pathlib import Path

def download_plotly():
    """Download the latest version of Plotly.js."""
    url = "https://cdn.plot.ly/plotly-2.27.1.min.js"
    output_dir = Path("static/js")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "plotly-2.27.1.min.js"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
            
        print(f"Plotly.js downloaded successfully to {output_file}")
    except Exception as e:
        print(f"Error downloading Plotly.js: {e}")

if __name__ == "__main__":
    download_plotly() 