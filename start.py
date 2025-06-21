#!/usr/bin/env python3
"""
Startup script for Stock Volatility Dashboard
Run this to start both backend and frontend
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'yfinance', 
        'streamlit', 'plotly', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing packages. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ Packages installed successfully!")
    else:
        print("✅ All packages are installed!")

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI backend...")
    try:
        subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"
        ])
        print("✅ Backend started on http://localhost:8000")
        print("📚 API docs available at http://localhost:8000/docs")
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")

def start_frontend():
    """Start the Streamlit frontend"""
    print("🎨 Starting Streamlit frontend...")
    time.sleep(3)  # Wait for backend to start
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501", "--server.address", "localhost"
        ])
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")

def main():
    print("📈 Stock Volatility Dashboard - Hackathon Edition")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Start backend
    start_backend()
    
    # Start frontend
    start_frontend()

if __name__ == "__main__":
    main() 