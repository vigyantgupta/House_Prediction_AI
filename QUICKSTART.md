# QUICK START - Run This!

## For Windows Users:
**Double-click:** `run.bat`

This will:
1. Create virtual environment automatically
2. Install all dependencies
3. Train the model (if needed)
4. Start the API server
5. Open your browser to the frontend

---

## For Mac/Linux Users:
```bash
python run.py
```

---

## Manual Setup (if scripts don't work):

```bash
# 1. Navigate to project
cd House_Prediction_AI

# 2. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
# OR
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model
python -m src.train

# 5. Start API
uvicorn api.main:app --reload

# 6. Open in browser
# http://localhost:8000/docs
```

---

## Using the Application:

1. **Predict Price Tab** - Enter house details to get price predictions
2. **Model Information Tab** - View model performance metrics
3. **Features Tab** - See all available features

---

## What Each Tab Does:

### Predict Price
- Fill in house details (lot area, overall quality, year built, etc.)
- Click "Predict Price"
- Get instant ML-powered price prediction

### Model Information
- Shows which ML model is being used
- Displays number of features
- Shows model file paths

### Features
- Lists all 80+ features the model expects
- Helps you understand what data the system uses

---

## Troubleshooting:

**"Error loading features"** - API server not running. Click "run.bat" or "run.py"

**Port 8000 in use** - Close other applications or use: `uvicorn api.main:app --port 8001`

**Model not found** - Run training: `python -m src.train`

---

That's it! The system is ready to use.
