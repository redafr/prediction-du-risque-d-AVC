import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Chargement des modèles et objets ===
model_path = r"C:\Users\Hp\Desktop\2025 - Reda Fritet - RAUC\archive (2)"

xgb = joblib.load(f"{model_path}\\xgboost_final_ctgan.pkl")
lgbm = joblib.load(f"{model_path}\\lightgbm_final_ctgan.pkl")
rf = joblib.load(f"{model_path}\\randomforest_final_ctgan.pkl")
scaler = joblib.load(f"{model_path}\\scaler.pkl")
encoders = joblib.load(f"{model_path}\\encoders.pkl")

# === Interface Streamlit ===
st.title(" Prédiction du risque d'AVC")

# === Saisie utilisateur ===
gender = st.selectbox("Genre", ["Male", "Female"])
age = st.slider("Âge", 1, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Maladie cardiaque", [0, 1])
ever_married = st.selectbox("Marié(e)", ["Yes", "No"])
work_type = st.selectbox("Type de travail", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Type de résidence", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Niveau moyen de glucose", 50.0, 300.0, 100.0)
bmi = st.number_input("IMC", 10.0, 60.0, 25.0)
smoking_status = st.selectbox("Tabagisme", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# === Transformation des données ===
input_dict = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status
}

df_input = pd.DataFrame([input_dict])

# Encodage des colonnes catégorielles
for col, encoder in encoders.items():
    df_input[col] = encoder.transform(df_input[col].astype(str))

# Normalisation
df_input[["age", "avg_glucose_level", "bmi"]] = scaler.transform(df_input[["age", "avg_glucose_level", "bmi"]])

# === Prédiction ===
model_choice = st.selectbox("Modèle", ["Random Forest"])

if st.button("Prédire le risque d'AVC"):
    if model_choice == "XGBoost":
        model = xgb
    elif model_choice == "LightGBM":
        model = lgbm
    else:
        model = rf

    proba = model.predict_proba(df_input)[0][1]
    prediction = model.predict(df_input)[0]

    st.subheader("Résultat de la prédiction")
    st.write(f"Probabilité estimée : **{proba:.2%}**")
