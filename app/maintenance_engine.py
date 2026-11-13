"""
Individual SHAP explanations and sensor-specific maintenance guidance.
This explains model behaviour; it does not diagnose a physical fault.
"""
import numpy as np
import pandas as pd
import shap

SENSOR_INSPECTION_GUIDE = {
    "engine rpm": "Verify the RPM sensor signal and wiring; compare the reading with the engine operating state and commanded speed/load.",
    "lub oil pressure": "Verify the oil-pressure sensor; check oil level/specification, filter condition, possible leaks, and lubrication-system operation.",
    "fuel pressure": "Verify the fuel-pressure sensor and connections; inspect fuel supply, filter, pump operation, and possible leakage.",
    "coolant pressure": "Verify the coolant-pressure sensor; inspect coolant level, hoses, radiator, cap, pump, and possible leaks.",
    "lub oil temp": "Verify the oil-temperature sensor; check oil level/specification and inspect lubrication and oil-cooling systems.",
    "coolant temp": "Verify the coolant-temperature sensor; inspect coolant level, radiator airflow, thermostat, cooling fan, and coolant pump.",
}

def _class_index(model, predicted_class):
    for index, label in enumerate(list(model.classes_)):
        if str(label) == str(predicted_class):
            return index
    raise ValueError(f"Predicted class {predicted_class!r} not found in model.classes_.")

def _select_class_values(raw_values, class_index, n_features):
    if isinstance(raw_values, list):
        raw_values = raw_values[class_index]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        if values.shape[0] == 1 and values.shape[1] == n_features:
            values = values[0, :, class_index]       # (samples, features, classes)
        elif values.shape[1] == 1 and values.shape[2] == n_features:
            values = values[class_index, 0, :]       # (classes, samples, features)
        else:
            raise ValueError(f"Unexpected 3-D SHAP shape: {values.shape}")
    elif values.ndim == 2:
        if values.shape[0] == 1 and values.shape[1] == n_features:
            values = values[0, :]
        elif values.shape[1] == 1 and values.shape[0] == n_features:
            values = values[:, 0]
        else:
            raise ValueError(f"Unexpected 2-D SHAP shape: {values.shape}")
    elif values.ndim != 1:
        raise ValueError(f"Unexpected SHAP dimensions: {values.shape}")
    if len(values) != n_features:
        raise ValueError(f"SHAP returned {len(values)} values for {n_features} features.")
    return values.astype(float)

def _select_base_value(raw_base, class_index):
    base = np.asarray(raw_base)
    if base.ndim == 0:
        return float(base)
    if base.ndim == 2:
        return float(base[0, class_index])
    if base.ndim == 1:
        return float(base[0] if len(base) == 1 else base[class_index])
    raise ValueError(f"Unexpected SHAP base-value shape: {base.shape}")

def explain_one_prediction(model, processed_input, feature_names, predicted_class, raw_sensor_values=None):
    """Explain the exact imputed row passed to model.predict()."""
    processed_input = np.asarray(processed_input)
    feature_names = list(feature_names)
    if processed_input.ndim != 2 or processed_input.shape[0] != 1:
        raise ValueError("Expected exactly one processed input row.")
    if processed_input.shape[1] != len(feature_names):
        raise ValueError("Feature count does not match metadata feature names.")

    class_index = _class_index(model, predicted_class)
    explainer = shap.TreeExplainer(model)
    shap_result = explainer(processed_input)
    values = _select_class_values(shap_result.values, class_index, len(feature_names))
    raw_base = getattr(shap_result, "base_values", None)
    if raw_base is None:
        raw_base = explainer.expected_value
    base_value = _select_base_value(raw_base, class_index)

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=processed_input[0],
        feature_names=feature_names,
    )

    raw_sensor_values = raw_sensor_values or {}
    rows = []
    for feature, shap_value in zip(feature_names, values):
        direction = (
            "Pushes model output toward predicted class" if shap_value > 0
            else "Pushes model output away from predicted class" if shap_value < 0
            else "No directional contribution"
        )
        rows.append({
            "Sensor": feature,
            "Current Reading": raw_sensor_values.get(feature, np.nan),
            "SHAP Contribution": float(shap_value),
            "Absolute Contribution": abs(float(shap_value)),
            "Direction": direction,
        })
    contribution_df = pd.DataFrame(rows).sort_values(
        "Absolute Contribution", ascending=False
    ).reset_index(drop=True)
    return explanation, contribution_df

def build_maintenance_recommendations(condition, contribution_df, top_n=3):
    """Generate inspection prompts from the highest-magnitude SHAP drivers."""
    condition = str(condition).strip().lower()
    if condition == "unhealthy":
        overall = (
            "The model predicts an unhealthy engine-condition label. Arrange a prompt "
            "inspection by a qualified technician and confirm with physical checks "
            "and OEM procedures."
        )
    elif condition == "healthy":
        overall = (
            "The model predicts a healthy engine-condition label. Continue the "
            "applicable scheduled maintenance programme and monitor sensor trends."
        )
    else:
        overall = (
            "The predicted label could not be interpreted. Review target-class "
            "mapping and sensor data before taking action."
        )

    rows = []
    for _, row in contribution_df.head(top_n).iterrows():
        sensor = str(row["Sensor"])
        action = SENSOR_INSPECTION_GUIDE.get(
            sensor.strip().lower(),
            "Verify this sensor's signal, wiring, units, and plausibility against "
            "the vehicle manufacturer's service information."
        )
        rows.append({
            "Sensor": sensor,
            "Current Reading": row["Current Reading"],
            "Model Contribution": round(float(row["SHAP Contribution"]), 6),
            "Suggested Inspection": action,
        })
    return overall, pd.DataFrame(rows)
