@echo off
REM House Price Prediction System - Auto Startup
REM This script installs dependencies, trains the model, starts the API, and opens the frontend

echo.
echo =========================================
echo House Price Prediction System Startup
echo =========================================
echo.

REM Check if venv exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Install dependencies silently
echo Installing dependencies...
pip install -q fastapi uvicorn scikit-learn pandas numpy pydantic python-dotenv matplotlib joblib 2>nul

REM Train model if best_model doesn't exist
if not exist "model\best_model.joblib" (
    echo.
    echo Training model (this will take 1-2 minutes)...
    python -m src.train
    echo Model training complete!
)

REM Start API in background
echo.
echo Starting API server...
start "" python -m uvicorn api.main:app --reload --port 8000

REM Wait for server to start
timeout /t 3 /nobreak

REM Open frontend in default browser
echo Opening frontend in browser...
start http://localhost:8000/docs

echo.
echo =========================================
echo System is running!
echo =========================================
echo.
echo API Documentation: http://localhost:8000/docs
echo Frontend: frontend/index.html (or open the browser window above)
echo.
echo Press Ctrl+C in this window to stop the server
echo.

REM Keep the window open
cmd /k
