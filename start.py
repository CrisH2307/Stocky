# start.py
import subprocess
import sys
import time

# Start FastAPI backend
backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--reload"])

# Wait a bit to ensure backend is up
time.sleep(2)

# Start Streamlit frontend
frontend = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard.py"])

# Wait for both processes to finish
try:
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    backend.terminate()
    frontend.terminate()