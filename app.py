import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# --- 1. LOAD AI MODELS ---
@st.cache_resource
def load_disease_model():
    # Your CNN model for leaf analysis
    return tf.keras.models.load_model('crop_disease_model.h5')

@st.cache_resource
def load_yield_model():
    # Your Random Forest model for harvest prediction
    return joblib.load('yield_model.pkl')

disease_model = load_disease_model()
yield_model = load_yield_model()

# Categories for Disease Detection
class_names = ['Healthy', 'Leaf Rust', 'Blight']

# --- 2. PAGE SETUP ---
st.set_page_config(page_title="Smart Agriculture System", layout="wide")
st.title("🌾 Smart Crop Monitoring & Yield Prediction")

# --- 3. TOP SECTION: LIVE SENSORS ---
st.subheader("📡 Live Environmental Data")
col1, col2, col3 = st.columns(3)

# For now, these are manual sliders to simulate your ESP32 sensors
# Once your hardware arrives, we will connect these to your real-time API
s_moist = col1.slider("💧 Soil Moisture (%)", 0, 100, 45)
s_temp = col2.slider("🌡️ Temperature (°C)", 10, 50, 28)
s_hum = col3.slider("☁️ Humidity (%)", 0, 100, 60)

st.divider()

# --- 4. MAIN DASHBOARD ---
left_col, right_col = st.columns(2)

# LEFT SIDE: DISEASE DETECTION
with left_col:
    st.header("🌿 Crop Disease Detection")
    st.write("Upload a leaf image for AI analysis.")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Image Preprocessing
        img_resized = image.resize((150, 150))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("Analyzing leaf..."):
            predictions = disease_model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
        
        if predicted_class == 'Healthy':
            st.success(f"🔬 Diagnosis: {predicted_class} ({confidence:.1f}%)")
        else:
            st.error(f"⚠️ Diagnosis: {predicted_class} ({confidence:.1f}%)")

# RIGHT SIDE: YIELD PREDICTION
with right_col:
    st.header("📈 Yield Prediction")
    st.write("Predicting harvest based on current sensor inputs.")
    
    # Prepare data for Random Forest model
    input_features = np.array([[s_moist, s_temp, s_hum]])
    
    with st.spinner("Calculating Yield..."):
        predicted_yield = yield_model.predict(input_features)[0]
    
    # Display Results
    st.metric(label="Estimated Harvest", value=f"{predicted_yield:.2f} kg/hectare")
    
    # Visual Progress Bar (Target Max: 4000 kg)
    progress_val = min(int(predicted_yield / 4000 * 100), 100)
    st.progress(progress_val)
    st.write(f"Efficiency: {progress_val}% of maximum capacity.")

    # Status Advice
    if s_moist < 30:
        st.warning("🚨 Low Moisture! Irrigation recommended to improve yield.")
    elif 40 <= s_moist <= 70:
        st.success("✅ Ideal conditions for maximum crop growth.")
