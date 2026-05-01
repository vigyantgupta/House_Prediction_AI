#!/usr/bin/env python
"""Auto-startup script for House Price Prediction System."""

import subprocess
import sys
import time
import os
import webbrowser
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    os.chdir(project_root)

    print("\n" + "="*50)
    print("House Price Prediction System - Auto Startup")
    print("="*50 + "\n")

    # Create venv if needed
    venv_path = project_root / ".venv"
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"])

    # Get Python executable from venv
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"

    # Install dependencies
    print("Installing dependencies...")
    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "-q",
         "fastapi", "uvicorn", "scikit-learn", "pandas", "numpy",
         "pydantic", "python-dotenv", "matplotlib", "joblib"],
        capture_output=True
    )

    # Train model if needed
    model_path = project_root / "model" / "best_model.joblib"
    if not model_path.exists():
        print("\nTraining model (this will take 1-2 minutes)...")
        subprocess.run([str(python_exe), "-m", "src.train"])
        print("Model training complete!\n")

    # Start API server
    print("Starting API server...")
    api_process = subprocess.Popen(
        [str(python_exe), "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000"],
        cwd=project_root
    )

    # Wait for server to start
    time.sleep(3)

    # Open browser
    print("Opening frontend in browser...\n")
    webbrowser.open("http://localhost:8000/docs")

    print("="*50)
    print("System is running!")
    print("="*50)
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Frontend HTML: frontend/index.html")
    print("\nPress Ctrl+C to stop the server\n")

    try:
        api_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_process.terminate()
        api_process.wait()
        print("Done!")

if __name__ == "__main__":
    main()
