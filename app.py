import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    model = pickle.load(open('crop_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_model()

st.set_page_config(page_title="Crop Recommendation", page_icon="🌱")
st.title("🌾 Crop Recommendation System")
st.markdown("Apne khet ke conditions daal kar best crop jaaniye!")

st.sidebar.header("Conditions Daaliye")

N = st.sidebar.slider('Nitrogen (N)', 0, 140, 70)
P = st.sidebar.slider('Phosphorus (P)', 5, 145, 50)
K = st.sidebar.slider('Potassium (K)', 5, 205, 40)
temperature = st.sidebar.slider('Temperature (°C)', 8.0, 45.0, 25.0)
humidity = st.sidebar.slider('Humidity (%)', 10.0, 100.0, 70.0)
ph = st.sidebar.slider('pH Value', 3.5, 9.9, 6.5)
rainfall = st.sidebar.slider('Rainfall (mm)', 20.0, 300.0, 150.0)

input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

st.subheader("Aapke Inputs")
st.write(f"N: **{N}** | P: **{P}** | K: **{K}**")
st.write(f"Temperature: **{temperature:.2f}°C** | Humidity: **{humidity:.2f}%**")
st.write(f"pH: **{ph:.2f}** | Rainfall: **{rainfall:.2f} mm**")

scaled_input = scaler.transform(input_data)
prediction = model.predict(scaled_input)[0]
probabilities = model.predict_proba(scaled_input)[0]

st.markdown("## 🏆 Sabse Best Crop")
st.success(f"**{prediction.upper()}**")

st.markdown("### Top 5 Suitable Crops (Confidence %)")
proba_df = pd.DataFrame({
    'Crop': model.classes_,
    'Confidence (%)': np.round(probabilities * 100, 2)
})
proba_df = proba_df.sort_values('Confidence (%)', ascending=False).head(5)
st.table(proba_df.reset_index(drop=True))

st.info("Model 2200+ real samples se trained hai!")