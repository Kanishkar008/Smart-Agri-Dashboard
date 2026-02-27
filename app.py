import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib

# --- 1. THEME & STYLING ---
st.set_page_config(page_title="AgroSmart AI", page_icon="🌿", layout="wide")

# Unique CSS for a high-end "Dark Mode" aesthetic
st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #0b0e14; }
    
    /* Custom Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: white;
    }
    .stTabs [aria-selected="true"] { background-color: #4CAF50 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_all_models():
    d_model = tf.keras.models.load_model('crop_disease_model.h5')
    y_model = joblib.load('yield_model.pkl')
    return d_model, y_model

disease_model, yield_model = load_all_models()
class_names = ['Healthy', 'Leaf Rust', 'Blight']

# --- 3. SIDEBAR BRANDING ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚜 AgroSmart</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Powered Precision Farming</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📡 Hardware Feed")
    # Once hardware arrives, we replace these sliders with API calls
    s_moist = st.slider("Soil Moisture (%)", 0, 100, 45)
    s_temp = st.slider("Temperature (°C)", 10, 50, 28)
    s_hum = st.slider("Humidity (%)", 0, 100, 60)
    st.divider()
    st.success("Cloud Sync: Active")

# --- 4. MAIN DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["📊 Live Insights", "🔬 AI Diagnosis", "📜 Project Info"])

with tab1:
    st.title("Field Command Center")
    
    # Top Row: Metrics
    m1, m2, m3 = st.columns(3)
    
    # Yield Prediction Logic
    input_data = np.array([[s_moist, s_temp, s_hum]])
    prediction = yield_model.predict(input_data)[0]
    
    m1.metric("Predicted Yield", f"{prediction:.1f} kg/ha", delta=f"{prediction-2000:.1f} vs Avg")
    m2.metric("Soil Status", "Optimal" if 40 <= s_moist <= 70 else "Check Required")
    m3.metric("Climate Score", "88/100")
    
    st.divider()
    
    # Visual Progress Section
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Yield Efficiency Gauge")
        # Visualizing the harvest capacity
        progress_val = min(int(prediction / 4000 * 100), 100)
        st.progress(progress_val)
        st.write(f"Current conditions are producing **{progress_val}%** of maximum theoretical yield.")
    
    with col_b:
        st.subheader("💡 System Advice")
        if s_moist < 35:
            st.error("🚨 CRITICAL: Trigger irrigation system.")
        elif s_temp > 35:
            st.warning("⚠️ ALERT: High heat detected. Check crop shade.")
        else:
            st.success("✅ STABLE: No immediate actions required.")

with tab2:
    st.title("AI Crop Disease Laboratory")
    st.write("Scan high-resolution leaf images to detect cellular abnormalities.")
    
    upload = st.file_uploader("Drop image here...", type=["jpg", "png", "jpeg"])
    
    if upload:
        img = Image.open(upload)
        # Display image in a styled container
        st.image(img, caption="Analyzed Sample", use_container_width=True)
        
        # AI Processing
        img_arr = np.array(img.resize((150, 150)))
        img_arr = np.expand_dims(img_arr, axis=0)
        
        with st.spinner("Decoding image patterns..."):
            pred = disease_model.predict(img_arr)
            result = class_names[np.argmax(pred)]
            conf = np.max(pred) * 100
            
        # Display Result in unique card
        st.markdown(f"""
            <div style='background-color:rgba(76, 175, 80, 0.1); padding:20px; border-radius:10px; border-left: 5px solid #4CAF50;'>
                <h3 style='margin:0;'>Result: {result}</h3>
                <p style='margin:0;'>Confidence Score: {conf:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

with tab3:
    st.header("Project Documentation")
    st.write("**Title:** AI-Based Crop Disease Detection & Yield Prediction")
    st.write("**Semester:** 6th Semester Capstone")
    st.write("**Tech Stack:** TensorFlow, Random Forest, ESP32, Streamlit Cloud")
