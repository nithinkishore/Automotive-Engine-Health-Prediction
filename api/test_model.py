
import joblib
import json
import os

# Project model directory
MODEL_DIR = "models"

# Load trained Random Forest
model = joblib.load(
    os.path.join(MODEL_DIR, "engine_health_model.joblib")
)

# Load imputer
imputer = joblib.load(
    os.path.join(MODEL_DIR, "engine_health_imputer.joblib")
)

# Load metadata
with open(
    os.path.join(MODEL_DIR, "engine_health_metadata.json"),
    "r"
) as file:
    metadata = json.load(file)

print("MODEL LOADED SUCCESSFULLY!")
print("Model:", model)
print("Features:", metadata["features"])
print("Classes:", metadata["classes"])
print("Number of features:", len(metadata["features"]))
