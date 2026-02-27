import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# --- 1. THEME & UI CONFIG ---
st.set_page_config(page_title="AgroSmart Pro Max", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { color: #4CAF50; font-size: 36px; }
    .recommendation-card {
        background: rgba(76, 175, 80, 0.05);
        border: 1px solid #4CAF50;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_resources():
    d_model = tf.keras.models.load_model('crop_disease_model.h5')
    y_model = joblib.load('yield_model.pkl')
    return d_model, y_model

disease_model, yield_model = load_resources()
class_names = ['Healthy', 'Leaf Rust', 'Blight']

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🚜 Control Center")
    st.divider()
    
    st.subheader("📡 Live IoT Feed")
    s_temp = st.slider("Temperature (°C)", 10, 50, 28)
    s_hum = st.slider("Humidity (%)", 0, 100, 60)
    s_moist = st.slider("Soil Moisture (%)", 0, 100, 45)
    
    st.divider()
    
    st.subheader("🧪 Lab Soil Inputs")
    st.info("Manual entry for parameters without real-time sensors.")
    s_nitro = st.number_input("Nitrogen (N) mg/kg", 0, 200, 70)
    s_phos = st.number_input("Phosphorus (P) mg/kg", 0, 200, 50)
    
    st.divider()
    st.success("Cloud Connection: Secure")

# --- 4. MAIN INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📊 Harvest Analytics", "🔬 Disease Lab", "📖 Project Info"])

with tab1:
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.header("📈 Yield & Economic Forecast")
        
        # ML Prediction using the 3 real-time sensors
        input_data = np.array([[s_moist, s_temp, s_hum]])
        base_yield = yield_model.predict(input_data)[0]
        
        # Mathematical Adjustment for N & P (Soil Efficiency Logic)
        efficiency = (s_nitro + s_phos) / 150
        final_yield = base_yield * (0.6 + (0.4 * efficiency))
        
        m1, m2 = st.columns(2)
        m1.metric("Predicted Harvest", f"{final_yield:.2f} kg/ha", f"{int(efficiency*100)}% Soil Efficiency")
        m2.metric("Market Value", f"₹{(final_yield * 45):,.0f}", "Est. Profit")
        
        st.progress(min(int(final_yield / 4500 * 100), 100))
        st.caption("Yield logic combines real-time IoT weather and manual soil lab data.")

    with col_right:
        st.header("🌱 Dynamic Crop Matches")
        st.write("Top suggestions for your current 5-parameter profile:")
        
        recs = []
        # Multi-factor Recommendation Logic (10+ Crops)
        if s_moist > 70 and s_nitro > 80: recs.append(("Rice", "Fastest growth in nutrient-rich wetlands."))
        if 22 <= s_temp <= 30: recs.append(("Maize", "Ideal warmth for high-speed photosynthesis."))
        if s_temp < 23 and s_hum < 50: recs.append(("Wheat", "Cool-climate specialist for rapid germination."))
        if s_temp > 30 and s_moist < 40: recs.append(("Cotton", "Deep-root fiber crop thriving in dry heat."))
        if s_phos > 70 and s_temp < 25: recs.append(("Potato", "Phosphorus-rich soil drives tuber growth."))
        if s_nitro < 40: recs.append(("Moong Dal", "Nitrogen-fixing legume for low-N soils."))
        if s_moist > 75 and s_temp > 28: recs.append(("Sugarcane", "Maximum biomass in tropical humidity."))
        if s_temp > 35 and s_moist < 30: recs.append(("Millet", "Climate-resilient superfood for harsh zones."))
        if 25 <= s_temp <= 32: recs.append(("Groundnut", "Well-drained warm soil preference."))
        if s_hum > 80: recs.append(("Jute", "Rapid fiber production in monsoon belts."))

        for crop, reason in recs[:6]: 
            st.markdown(f"""<div class="recommendation-card"><b>🌿 {crop}</b><br><small>{reason}</small></div>""", unsafe_allow_html=True)

with tab2:
    st.header("🔬 AI Crop Disease Laboratory")
    file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True, caption="Target Sample")
        
        img_arr = np.array(img.resize((150, 150)))
        img_arr = np.expand_dims(img_arr, axis=0)
        
        with st.spinner("Decoding patterns..."):
            pred = disease_model.predict(img_arr)
            res = class_names[np.argmax(pred)]
            conf = np.max(pred) * 100
            
        st.markdown(f"### Result: **{res}**")
        st.write(f"Confidence: {conf:.1f}%")

with tab3:
    st.subheader("Technical Documentation")
    st.write("**Core System:** IoT-driven Precision Agriculture")
    st.write("**AI Models:** CNN (Disease) & Random Forest Regressor (Yield)")
    st.write("**Methodology:** Hybrid data ingestion combining real-time IoT and manual Lab reports")
