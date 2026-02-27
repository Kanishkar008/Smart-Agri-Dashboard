import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load your saved AI Brain
@st.cache_resource
def load_model():
    # This loads the exact .h5 file you built on Day 1!
    return tf.keras.models.load_model('crop_disease_model.h5')

model = load_model()
# These are the 3 categories we told the AI to learn in Phase 2
class_names = ['Healthy', 'Leaf Rust', 'Blight'] 

# Setup the page layout
st.set_page_config(page_title="Smart Agriculture System", layout="wide")
st.title("🌾 Smart Crop Monitoring & Yield Prediction")

# Top Section: Field Sensors
st.subheader("📡 Live Environmental Data")
col1, col2, col3 = st.columns(3)
col1.metric(label="💧 Soil Moisture", value="45%", delta="-2%")
col2.metric(label="🌡️ Temperature", value="28 °C", delta="+1 °C")
col3.metric(label="☁️ Humidity", value="60%", delta="0%")

st.divider()

left_col, right_col = st.columns(2)

# Left Half: Disease Detection (NOW LIVE!)
with left_col:
    st.header("🌿 Crop Disease Detection")
    st.write("Upload a leaf image to run it through your CNN.")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess the image so the AI can read it (resize to 150x150)
        img_resized = image.resize((150, 150))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) 
        
        # Make the actual prediction!
        with st.spinner("AI is analyzing the leaf..."):
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
        
        # Show the result on the screen
        if predicted_class == 'Healthy':
            st.success(f"🔬 AI Diagnosis: {predicted_class} ({confidence:.1f}% confidence)")
        else:
            st.error(f"⚠️ AI Diagnosis: {predicted_class} ({confidence:.1f}% confidence)")

# Right Half: Yield Prediction
with right_col:
    st.header("📈 Yield Prediction")
    st.write("Using Random Forest Machine Learning.")
    st.write("Based on the current 28°C temp and 45% moisture:")
    
    st.info("🌾 Estimated Harvest: 2,400 kg/hectare")
    st.progress(75)
