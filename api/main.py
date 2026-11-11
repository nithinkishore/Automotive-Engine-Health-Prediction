
from pathlib import Path
import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------
# 1. Load model artifacts
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

model = joblib.load(
    MODEL_DIR / "engine_health_model.joblib"
)

imputer = joblib.load(
    MODEL_DIR / "engine_health_imputer.joblib"
)

with open(MODEL_DIR / "engine_health_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

# --------------------------------------------------
# 2. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Automotive Engine Health Prediction API",
    description="ML-based engine condition prediction using sensor data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------------------------------------------
# 3. Define sensor input
# --------------------------------------------------

class EngineSensorInput(BaseModel):
    engine_rpm: float
    lub_oil_pressure: float
    fuel_pressure: float
    coolant_pressure: float
    lub_oil_temp: float
    coolant_temp: float

# --------------------------------------------------
# 4. API home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Automotive Engine Health API is running",
        "model": "Random Forest",
        "features": FEATURES
    }

# --------------------------------------------------
# 5. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }

# --------------------------------------------------
# 6. Engine prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_engine_condition(data: EngineSensorInput):

    try:
        sensor_values = {
            "Engine rpm": data.engine_rpm,
            "Lub oil pressure": data.lub_oil_pressure,
            "Fuel pressure": data.fuel_pressure,
            "Coolant pressure": data.coolant_pressure,
            "lub oil temp": data.lub_oil_temp,
            "Coolant temp": data.coolant_temp
        }

        input_df = pd.DataFrame(
            [[sensor_values[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        # Apply the same imputer used during training
        processed_input = imputer.transform(input_df)

        # Generate prediction
        prediction = model.predict(processed_input)[0]

        # Calculate class probabilities
        probabilities = model.predict_proba(processed_input)[0]

        probability_results = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(model.classes_, probabilities)
        }

        return {
            "predicted_class": str(prediction),
            "class_probabilities": probability_results,
            "prediction_confidence": round(
                float(max(probabilities)), 4
            ),
            "model": "Random Forest"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
