import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("C:/AppBC/model_bc_randomforest.pkl")
scaler = joblib.load("C:/AppBC/scaler_bc.pkl")

# Judul
st.set_page_config(page_title="Deteksi Kanker Payudara", layout="centered")
st.title("🩺 Deteksi Dini Kanker Payudara")
st.write("Masukkan nilai-nilai karakteristik sel histopatologi untuk prediksi keganasan.")

# Fitur input
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

user_input = []
for feature in features:
    value = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0, value=10.0, step=0.1)
    user_input.append(value)

# Prediksi
if st.button("Prediksi"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)[0]
    label = "💖 Jinak (Benign)" if pred == 0 else "⚠️ Ganas (Malignant)"
    st.subheader("Hasil Prediksi:")
    st.success(label)
