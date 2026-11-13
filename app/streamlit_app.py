
import shap
import matplotlib.pyplot as plt

from maintenance_engine import (
    explain_one_prediction,
    build_maintenance_recommendations
)
from pathlib import Path
from datetime import datetime
import json
import joblib

import streamlit as st
import pandas as pd
import plotly.express as px
# Prediction history storage
PROJECT_DIR = Path(__file__).resolve().parent.parent

HISTORY_FILE = PROJECT_DIR / "data" / "prediction_history.csv"

HISTORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR = PROJECT_DIR / "models"


@st.cache_resource
def load_model_artifacts():
    """Load the trained model and preprocessing artifacts once."""
    model = joblib.load(MODEL_DIR / "engine_health_model.joblib")
    imputer = joblib.load(MODEL_DIR / "engine_health_imputer.joblib")
    with open(MODEL_DIR / "engine_health_metadata.json", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, imputer, metadata["features"]

st.set_page_config(
    page_title="Automotive Engine Health AI",
    page_icon="🚗",
    layout="wide"
)

# --------------------------------------------------
# 2. DASHBOARD HEADER
# --------------------------------------------------

st.title("🚗 Automotive Engine Health Prediction")
st.caption(
    "Machine Learning | Explainable AI | Predictive Maintenance"
)

st.markdown(
    "Predict engine condition using six automotive sensor measurements "
    "powered by a Random Forest classification model."
)

st.divider()

# --------------------------------------------------
# 3. SENSOR INPUT FORM
# --------------------------------------------------

st.subheader("🔧 Engine Sensor Measurements")

with st.form("engine_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        engine_rpm = st.number_input(
            "Engine RPM",
            value=1500.0,
            step=100.0
        )

        lub_oil_pressure = st.number_input(
            "Lub Oil Pressure",
            value=3.5,
            step=0.1
        )

    with col2:
        fuel_pressure = st.number_input(
            "Fuel Pressure",
            value=11.0,
            step=0.5
        )

        coolant_pressure = st.number_input(
            "Coolant Pressure",
            value=2.5,
            step=0.1
        )

    with col3:
        lub_oil_temp = st.number_input(
            "Lub Oil Temperature",
            value=90.0,
            step=1.0
        )

        coolant_temp = st.number_input(
            "Coolant Temperature",
            value=85.0,
            step=1.0
        )

    predict_button = st.form_submit_button(
        "🔍 Predict Engine Health",
        width="stretch"
    )

# --------------------------------------------------
# 4. RUN LOCAL MODEL PREDICTION (NO FASTAPI REQUIRED)
# --------------------------------------------------

if predict_button:

    sensor_data = {
        "engine_rpm": engine_rpm,
        "lub_oil_pressure": lub_oil_pressure,
        "fuel_pressure": fuel_pressure,
        "coolant_pressure": coolant_pressure,
        "lub_oil_temp": lub_oil_temp,
        "coolant_temp": coolant_temp
    }

    try:
        with st.spinner("Loading model and analyzing engine sensor data..."):
            model, imputer, features = load_model_artifacts()

            # Match the exact feature names and order used during training.
            sensor_values = {
                "Engine rpm": engine_rpm,
                "Lub oil pressure": lub_oil_pressure,
                "Fuel pressure": fuel_pressure,
                "Coolant pressure": coolant_pressure,
                "lub oil temp": lub_oil_temp,
                "Coolant temp": coolant_temp
            }

            input_df = pd.DataFrame(
                [[sensor_values[feature] for feature in features]],
                columns=features
            )
            processed_input = imputer.transform(input_df)
            prediction = model.predict(processed_input)[0]
            model_probabilities = model.predict_proba(processed_input)[0]

            probabilities = {
                str(class_label): float(probability)
                for class_label, probability in zip(
                    model.classes_, model_probabilities
                )
            }

            result = {
                "predicted_class": str(prediction),
                "class_probabilities": probabilities,
                "prediction_confidence": float(max(model_probabilities)),
                "model": "Random Forest"
            }

        predicted_class = str(result["predicted_class"])
        probabilities = result["class_probabilities"]
        unhealthy_probability = float(probabilities["0"])
        healthy_probability = float(probabilities["1"])
        confidence = float(result["prediction_confidence"])

        # ------------------------------------------
        # 5. DISPLAY PREDICTION
        # ------------------------------------------

        st.divider()
        st.subheader("📊 Engine Health Prediction Result")

        if predicted_class == "0":
            st.error("⚠️ UNHEALTHY ENGINE CONDITION DETECTED")
            st.warning(
                "The model classified this sensor reading "
                "as an unhealthy engine condition. "
                "Further inspection is recommended."
            )
            condition = "Unhealthy"

        elif predicted_class == "1":
            st.success("✅ HEALTHY ENGINE CONDITION")
            st.info(
                "The model classified this sensor reading "
                "as a healthy engine condition."
            )
            condition = "Healthy"

        else:
            st.warning(f"Unexpected predicted class: {predicted_class}")
            condition = "Unknown"


        # --------------------------------------------------
        # INDIVIDUAL SHAP EXPLANATION
        # --------------------------------------------------

        contribution_df = pd.DataFrame()
        recommendation_df = pd.DataFrame()

        maintenance_summary = (
            "SHAP explanation unavailable. "
            "Manual inspection is recommended."
        )

        try:

            with st.spinner(
                "Generating individual SHAP explanation..."
            ):

                shap_explanation, contribution_df = explain_one_prediction(
                    model=model,
                    processed_input=processed_input,
                    feature_names=features,
                    predicted_class=prediction,
                    raw_sensor_values=sensor_values
                )

                # Generate maintenance recommendations
                maintenance_summary, recommendation_df = (
                    build_maintenance_recommendations(
                        condition=condition,
                        contribution_df=contribution_df,
                        top_n=3
                    )
                )

            st.divider()

            st.header("🧠 Individual SHAP Explanation")

            st.write(
                "Understand how each engine sensor measurement "
                "influenced this individual prediction."
            )

            # ------------------------------------------
            # SHAP WATERFALL CHART
            # ------------------------------------------

            st.subheader("Individual SHAP Waterfall Plot")

            shap.plots.waterfall(
                shap_explanation,
                max_display=len(features),
                show=False
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True
            )

            plt.close("all")

            # ------------------------------------------
            # SHAP CONTRIBUTION TABLE
            # ------------------------------------------

            st.subheader("Individual Sensor Contributions")

            st.dataframe(
                contribution_df[
                    [
                        "Sensor",
                        "Current Reading",
                        "SHAP Contribution",
                        "Direction"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------
            # SHAP CONTRIBUTION BAR CHART
            # ------------------------------------------

            contribution_chart = px.bar(
                contribution_df.sort_values(
                    "SHAP Contribution"
                ),
                x="SHAP Contribution",
                y="Sensor",
                orientation="h",
                color="Direction",
                title="Individual Sensor Contribution"
            )

            st.plotly_chart(
                contribution_chart,
                use_container_width=True
            )
            # ------------------------------------------
            # AI-BASED PREDICTIVE MAINTENANCE
            # ------------------------------------------

            st.divider()

            st.subheader("🔧 AI-Based Predictive Maintenance")

            st.info(maintenance_summary)

            if not recommendation_df.empty:
                st.write("Recommended sensor inspections:")
                st.dataframe(
                    recommendation_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning(
                    "No sensor-specific recommendations generated."
                )

            st.caption(
                "Recommendations are decision-support suggestions "
                "based on SHAP contributions, not confirmed fault diagnoses."
            )

        except Exception as shap_error:
            st.warning(
                "Individual SHAP explanation could not "
                f"be generated: {shap_error}"
            )


            

            

        # ------------------------------------------
        # 6. DISPLAY KPI METRICS
        # ------------------------------------------

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Predicted Condition", condition)
        metric2.metric("Prediction Confidence", f"{confidence * 100:.1f}%")
        metric3.metric("ML Model", result.get("model", "Random Forest"))

        # ------------------------------------------
        # 7. PROBABILITY VISUALIZATION
        # ------------------------------------------

        st.subheader("Prediction Probability")
        probability_df = pd.DataFrame({
            "Engine Condition": ["Unhealthy", "Healthy"],
            "Probability": [
                unhealthy_probability * 100,
                healthy_probability * 100
            ]
        })

        fig = px.bar(
            probability_df,
            x="Engine Condition",
            y="Probability",
            color="Engine Condition",
            text_auto=".1f",
            range_y=[0, 100],
            title="Engine Health Classification Probability"
        )
        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_title="Engine Condition"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------
        # 8. SENSOR MEASUREMENTS
        # ------------------------------------------

        st.subheader("Sensor Measurement Summary")
        sensor_df = pd.DataFrame({
            "Sensor": list(sensor_data.keys()),
            "Measured Value": list(sensor_data.values())
        })
        st.dataframe(
            sensor_df,
            width="stretch",
            hide_index=True
        )

        # ------------------------------------------
        # 9. SAVE PREDICTION HISTORY
        # ------------------------------------------

        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

        history_record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **sensor_data,
            "Predicted Condition": condition,
            "Confidence (%)": round(confidence * 100, 2)
        }
        st.session_state.prediction_history.append(history_record)

        history_df_new = pd.DataFrame([history_record])
        history_df_new.to_csv(
            HISTORY_FILE,
            mode="a",
            header=not HISTORY_FILE.exists(),
            index=False
        )
        st.success("Prediction saved to history!")

    except FileNotFoundError as error:
        st.error(
            "A required model artifact could not be found. "
            "Check that the models folder contains the model, imputer "
            "and metadata files. Details: " + str(error)
        )
    except Exception as error:
        st.error(f"Local model prediction failed: {error}")


# --------------------------------------------------
# 10. SHAP FEATURE IMPORTANCE REPORT
# --------------------------------------------------

st.divider()

st.header("🧠 Explainable AI – SHAP Analysis")

st.markdown(
    "Understand which sensor variables had the greatest "
    "overall influence on the trained model."
)

shap_csv = REPORT_DIR / "engine_health_shap_report.csv"

if shap_csv.exists():

    shap_df = pd.read_csv(shap_csv)

    st.subheader("Global SHAP Feature Importance")

    fig_shap = px.bar(
        shap_df.sort_values(
            "Mean_SHAP_Importance",
            ascending=True
        ),
        x="Mean_SHAP_Importance",
        y="Sensor",
        orientation="h",
        title="Mean Absolute SHAP Importance"
    )

    st.plotly_chart(
        fig_shap,
        width="stretch"
    )

else:

    st.info(
        "SHAP feature importance CSV is not available."
    )

# --------------------------------------------------
# 11. SHAP VISUAL REPORTS
# --------------------------------------------------

shap_image = REPORT_DIR / "shap_feature_importance.png"

rpm_image = REPORT_DIR / "engine_rpm_shap.png"

col1, col2 = st.columns(2)

with col1:

    if shap_image.exists():

        st.subheader("SHAP Feature Summary")

        st.image(
            str(shap_image),
            width="stretch"
        )

with col2:

    if rpm_image.exists():

        st.subheader("Engine RPM SHAP Analysis")

        st.image(
            str(rpm_image),
            width="stretch"
        )

# --------------------------------------------------
# 12. PREDICTION HISTORY
# --------------------------------------------------

st.divider()

st.header("📋 Prediction History")

# --------------------------------------------------
# 12. PREDICTION HISTORY ANALYTICS
# --------------------------------------------------

st.divider()

st.header("📋 Prediction History & Analytics")

# Load permanent prediction history
if HISTORY_FILE.exists():
    history_df = pd.read_csv(HISTORY_FILE)

elif st.session_state.get("prediction_history"):
    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

else:
    history_df = pd.DataFrame()

if not history_df.empty:

    # Identify prediction columns
    condition_col = next(
        (c for c in history_df.columns
         if "predicted" in c.lower()),
        None
    )

    confidence_col = next(
        (c for c in history_df.columns
         if "confidence" in c.lower()),
        None
    )

    # Calculate KPIs
    total_predictions = len(history_df)

    if condition_col:
        conditions = history_df[condition_col].astype(str).str.lower()

        unhealthy_count = conditions.str.contains(
            "unhealthy"
        ).sum()

        healthy_count = conditions.str.contains(
            "healthy"
        ).sum() - unhealthy_count
    else:
        unhealthy_count = 0
        healthy_count = 0

    if confidence_col:
        confidence_values = pd.to_numeric(
            history_df[confidence_col],
            errors="coerce"
        )

        average_confidence = confidence_values.mean()
    else:
        average_confidence = 0

    # KPI Dashboard
    st.subheader("📊 Prediction Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Predictions",
        total_predictions
    )

    col2.metric(
        "Unhealthy Engines",
        int(unhealthy_count)
    )

    col3.metric(
        "Healthy Engines",
        int(healthy_count)
    )

    col4.metric(
        "Average Confidence",
        f"{average_confidence:.1f}%"
    )

    st.divider()

    # Prediction Distribution
    if condition_col:
        st.subheader("Engine Health Distribution")

        distribution = history_df[
            condition_col
        ].value_counts().reset_index()

        distribution.columns = [
            "Engine Condition",
            "Count"
        ]

        fig = px.pie(
            distribution,
            names="Engine Condition",
            values="Count",
            hole=0.45,
            title="Healthy vs Unhealthy Predictions"
        )

        st.plotly_chart(fig, width="stretch")

    st.divider()

    # Prediction History Table
    st.subheader("📑 Historical Prediction Records")

    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True
    )

    # Download History
    csv_data = history_df.to_csv(index=False)

    st.download_button(
        "⬇️ Download Prediction History",
        data=csv_data,
        file_name="engine_prediction_history.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Prediction history will appear here "
        "after your first prediction."
    )

# --------------------------------------------------
# 12. SHAP EXPLAINABLE AI DASHBOARD
# --------------------------------------------------

import os
import pandas as pd

st.divider()

st.header("🔍 Explainable AI - SHAP Analysis")

st.write(
    "Understand which engine sensor measurements "
    "influence the Random Forest prediction."
)

# Project reports directory
PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")

# 1. Global SHAP Feature Importance
st.subheader("1. Sensor Feature Importance")

importance_file = os.path.join(
    REPORT_DIR,
    "engine_health_shap_report.csv"
)

if os.path.exists(importance_file):

    shap_importance = pd.read_csv(importance_file)

    shap_importance = shap_importance.sort_values(
        "Mean_SHAP_Importance",
        ascending=False
    )

    st.dataframe(
        shap_importance,
        width="stretch",
        hide_index=True
    )

else:
    st.warning("SHAP importance report not found.")

# 2. SHAP Feature Importance Chart
st.subheader("2. SHAP Feature Importance Visualization")

importance_image = os.path.join(
    REPORT_DIR,
    "shap_feature_importance.png"
)

if os.path.exists(importance_image):

    st.image(
        importance_image,
        caption="Global SHAP Feature Importance",
        width="stretch"
    )

else:
    st.warning("SHAP feature importance image not found.")

# 3. Engine RPM SHAP Dependence Plot
st.subheader("3. Engine RPM SHAP Dependence Analysis")

rpm_image = os.path.join(
    REPORT_DIR,
    "engine_rpm_shap.png"
)

if os.path.exists(rpm_image):

    st.image(
        rpm_image,
        caption="SHAP Dependence Plot - Engine RPM",
        width="stretch"
    )

else:
    st.warning("Engine RPM SHAP image not found.")

st.info(
    "SHAP explains how sensor measurements influence "
    "the machine learning model's predictions. "
    "These explanations describe model behaviour, "
    "not necessarily physical causation."
)

# --------------------------------------------------
# 13. ENGINE SENSOR TREND ANALYSIS
# --------------------------------------------------

st.divider()
st.header("📈 Engine Sensor Trend Analysis")

if HISTORY_FILE.exists():

    trend_df = pd.read_csv(HISTORY_FILE)

    sensor_columns = [
        "engine_rpm",
        "lub_oil_pressure",
        "fuel_pressure",
        "coolant_pressure",
        "lub_oil_temp",
        "coolant_temp"
    ]

    available_sensors = [
        col for col in sensor_columns
        if col in trend_df.columns
    ]

    if available_sensors:

        selected_sensor = st.selectbox(
            "Select Engine Sensor",
            available_sensors
        )

        trend_df[selected_sensor] = pd.to_numeric(
            trend_df[selected_sensor],
            errors="coerce"
        )

        trend_df["Prediction Number"] = (
            range(1, len(trend_df) + 1)
        )

        fig = px.line(
            trend_df,
            x="Prediction Number",
            y=selected_sensor,
            markers=True,
            title=f"{selected_sensor} Trend"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.info(
            "This chart displays sensor readings "
            "across historical predictions."
        )

    else:
        st.warning("No sensor columns found.")

else:
    st.info("No prediction history available.")

# --------------------------------------------------
# 14. PREDICTIVE MAINTENANCE INSIGHTS
# --------------------------------------------------

st.divider()

st.header("🔧 Predictive Maintenance Insights")

if HISTORY_FILE.exists():

    maintenance_df = pd.read_csv(HISTORY_FILE)

    if not maintenance_df.empty:

        # ------------------------------------------
        # 1. GET LATEST PREDICTION
        # ------------------------------------------

        latest = maintenance_df.iloc[-1]

        condition_col = next(
            (c for c in maintenance_df.columns
             if "predicted" in c.lower()),
            None
        )

        confidence_col = next(
            (c for c in maintenance_df.columns
             if "confidence" in c.lower()),
            None
        )

        if condition_col is None:
            st.error("Prediction condition column not found.")

        else:

            raw_condition = str(latest[condition_col]).strip().lower()

            if raw_condition in ["0", "0.0", "unhealthy"]:
                condition = "Unhealthy"
            elif raw_condition in ["1", "1.0", "healthy"]:
                condition = "Healthy"
            else:
                condition = "Unknown"

            confidence = (
                pd.to_numeric(
                    latest[confidence_col],
                    errors="coerce"
                )
                if confidence_col else 0
            )

            confidence = (
                0 if pd.isna(confidence) else float(confidence)
            )

            # --------------------------------------
            # 2. ENGINE RISK ASSESSMENT
            # --------------------------------------

            if condition == "Unhealthy":
                risk_level = "HIGH"
                priority = "P1"
                risk_description = (
                    "Unhealthy engine prediction. "
                    "Prompt inspection is recommended."
                )

            elif condition == "Healthy" and confidence < 60:
                risk_level = "MEDIUM"
                priority = "P2"
                risk_description = (
                    "Healthy prediction with relatively "
                    "low model confidence. Verify readings."
                )

            elif condition == "Healthy":
                risk_level = "LOW"
                priority = "P3"
                risk_description = (
                    "Model predicts healthy engine condition. "
                    "Continue routine monitoring."
                )

            else:
                risk_level = "UNKNOWN"
                priority = "REVIEW"
                risk_description = (
                    "Unable to interpret the latest prediction."
                )

            st.subheader("Engine Risk Assessment")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Engine Condition",
                condition
            )

            col2.metric(
                "Risk Level",
                risk_level
            )

            col3.metric(
                "Maintenance Priority",
                priority
            )

            if risk_level == "HIGH":
                st.error(risk_description)

            elif risk_level == "MEDIUM":
                st.warning(risk_description)

            elif risk_level == "LOW":
                st.success(risk_description)

            else:
                st.warning(risk_description)

            st.divider()

            # --------------------------------------
            # 3. SENSOR ANOMALY SCREENING
            # --------------------------------------

            st.subheader("Sensor Anomaly Screening")

            sensor_columns = [
                "engine_rpm",
                "lub_oil_pressure",
                "fuel_pressure",
                "coolant_pressure",
                "lub_oil_temp",
                "coolant_temp"
            ]

            available_sensors = [
                c for c in sensor_columns
                if c in maintenance_df.columns
            ]

            anomaly_results = []

            # Exclude latest record from baseline
            baseline = maintenance_df.iloc[:-1]

            if len(baseline) >= 5:

                for sensor in available_sensors:

                    historical = pd.to_numeric(
                        baseline[sensor],
                        errors="coerce"
                    ).dropna()

                    current_value = pd.to_numeric(
                        pd.Series([latest[sensor]]),
                        errors="coerce"
                    ).iloc[0]

                    if len(historical) < 5 or pd.isna(current_value):
                        continue

                    q1 = historical.quantile(0.25)
                    q3 = historical.quantile(0.75)
                    iqr = q3 - q1

                    if iqr <= 0:
                        continue

                    lower_limit = q1 - (1.5 * iqr)
                    upper_limit = q3 + (1.5 * iqr)

                    status = (
                        "Potential Anomaly"
                        if current_value < lower_limit
                        or current_value > upper_limit
                        else "Within Historical Range"
                    )

                    anomaly_results.append({
                        "Sensor": sensor,
                        "Current Reading": round(
                            current_value, 2
                        ),
                        "Historical Lower Bound": round(
                            lower_limit, 2
                        ),
                        "Historical Upper Bound": round(
                            upper_limit, 2
                        ),
                        "Status": status
                    })

                if anomaly_results:

                    anomaly_df = pd.DataFrame(
                        anomaly_results
                    )

                    st.dataframe(
                        anomaly_df,
                        width="stretch",
                        hide_index=True
                    )

                    anomaly_count = sum(
                        r["Status"] == "Potential Anomaly"
                        for r in anomaly_results
                    )

                    st.metric(
                        "Potential Sensor Anomalies",
                        anomaly_count
                    )

                else:
                    st.info(
                        "Insufficient variation in historical "
                        "sensor readings for anomaly screening."
                    )

            else:
                st.info(
                    f"At least 5 previous predictions are "
                    f"required for historical anomaly screening. "
                    f"Currently available: {len(baseline)}."
                )

            st.divider()

            # --------------------------------------
            # 4. MAINTENANCE RECOMMENDATIONS
            # --------------------------------------

            st.subheader("Maintenance Recommendations")

            if condition == "Unhealthy":

                recommendations = [
                    "Inspect the engine lubrication system.",
                    "Check fuel pressure and fuel delivery.",
                    "Inspect coolant pressure and cooling system.",
                    "Review engine operating temperature.",
                    "Perform physical engine inspection.",
                    "Validate sensor readings before repair decisions."
                ]

            elif condition == "Healthy":

                recommendations = [
                    "Continue scheduled preventive maintenance.",
                    "Monitor engine sensor measurements.",
                    "Inspect lubrication and cooling systems routinely.",
                    "Maintain historical sensor records."
                ]

            else:

                recommendations = [
                    "Review model output and input sensor readings.",
                    "Verify data quality before maintenance decisions."
                ]

            for item in recommendations:
                st.markdown(f"- {item}")

            st.divider()

            # --------------------------------------
            # 5. MAINTENANCE SUMMARY
            # --------------------------------------

            st.subheader("Maintenance Action Summary")

            summary = pd.DataFrame({
                "Parameter": [
                    "Latest Engine Condition",
                    "Prediction Confidence",
                    "Risk Assessment",
                    "Maintenance Priority",
                    "Recommended Action"
                ],
                "Result": [
                    condition,
                    f"{confidence:.2f}%",
                    risk_level,
                    priority,
                    (
                        "Prompt inspection recommended"
                        if condition == "Unhealthy"
                        else "Continue routine monitoring"
                        if condition == "Healthy"
                        else "Manual review required"
                    )
                ]
            })

            st.dataframe(
                summary,
                width="stretch",
                hide_index=True
            )

            st.caption(
                "Decision-support output based on model "
                "predictions and historical sensor readings. "
                "Not a substitute for physical inspection."
            )

else:

    st.info(
        "No prediction history available. "
        "Run an engine prediction first."
    )


# --------------------------------------------------
# 15. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Automotive Engine Health Prediction | "
    "Random Forest | FastAPI | Streamlit | SHAP"
)

st.caption(
    "Developed by Nithin Kishore R S"
)
