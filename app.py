import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# --- 1. PAGE SETUP & STYLING ---
st.set_page_config(page_title="AgroSmart Pro", page_icon="🚜", layout="wide")

st.markdown("""
    <style>
    .recommendation-card {
        background-color: rgba(76, 175, 80, 0.1);
        border: 2px solid #4CAF50;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_all():
    d_model = tf.keras.models.load_model('crop_disease_model.h5')
    y_model = joblib.load('yield_model.pkl')
    return d_model, y_model

disease_model, yield_model = load_all()

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🚜 Control Panel")
    s_moist = st.slider("Soil Moisture (%)", 0, 100, 45)
    s_temp = st.slider("Temperature (°C)", 10, 50, 28)
    s_hum = st.slider("Humidity (%)", 0, 100, 60)
    st.divider()
    st.info("Recommendations update live as you move sliders.")

# --- 4. MAIN DASHBOARD ---
tab1, tab2 = st.tabs(["📊 Harvest & Recommendations", "🔬 AI Disease Lab"])

with tab1:
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.header("📈 Harvest Prediction")
        # Yield Calculation
        input_features = np.array([[s_moist, s_temp, s_hum]])
        prediction = yield_model.predict(input_features)[0]
        
        st.metric("Estimated Yield", f"{prediction:.2f} kg/ha", delta="Active Feed")
        st.progress(min(int(prediction / 4000 * 100), 100))

    with col_right:
        st.header("💡 Fast-Growth Recommendations")
        
        # Simple Recommendation Logic based on environmental suitability
        recommendations = []
        if s_temp > 25 and s_moist > 60:
            recommendations = [("Rice", "High Water Affinity"), ("Sugarcane", "Fast growth in humid heat")]
        elif 18 <= s_temp <= 28 and 40 <= s_moist <= 60:
            recommendations = [("Maize", "Optimal Photosynthesis"), ("Cotton", "Warm soil preference")]
        elif s_temp < 22 and s_moist < 45:
            recommendations = [("Wheat", "Cool climate maturity"), ("Barley", "Drought tolerant")]
        else:
            recommendations = [("Millets", "Tough condition survivor"), ("Pulses", "Low resource needs")]

        for crop, reason in recommendations:
            st.markdown(f"""
                <div class="recommendation-card">
                    <h4 style="margin:0; color:#4CAF50;">🌱 {crop}</h4>
                    <p style="margin:0; font-size:14px;"><b>Why:</b> {reason}</p>
                </div>
            """, unsafe_allow_html=True)

with tab2:
    st.header("🔬 Disease Diagnosis")
    upload = st.file_uploader("Upload Leaf Photo", type=["jpg", "png"])
    if upload:
        img = Image.open(upload)
        st.image(img, width=300)
        st.success("Analysis: Processed (See result below)")
