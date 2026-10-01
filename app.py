import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

MODEL_PATH = 'brain_tumor_model.h5'
MODEL_URL = 'https://github.com/Skarif29/Brain-Tumor-Detection-Using-Deep-learning/releases/download/v1.0/Brain_tumor_model.h5'

st.title("Brain Tumor Detection App")

# --- MODEL DOWNLOAD LOGIC ---
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading trained model file... Please wait."):
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(MODEL_URL, headers=headers, stream=True, allow_redirects=True)
        
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            st.success("Model downloaded successfully!")
            st.rerun()
        else:
            st.error(f"Failed to download model. Status code: {response.status_code}")

# Dataset-er alphabetical class labels
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

@st.cache_resource
def load_model_dynamically():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    raise FileNotFoundError("Model file missing or corrupted.")

try:
    if os.path.exists(MODEL_PATH):
        model = load_model_dynamically()
        st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Upload an MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    img = image.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Predict"):
        if 'model' in locals():
            with st.spinner("Classifying..."):
                prediction = model.predict(img_array)
                
                scores = prediction[0]
                predicted_class_idx = np.argmax(scores)
                predicted_label = class_names[predicted_class_idx]
                confidence = float(scores[predicted_class_idx]) * 100
                
                st.write("---")
                st.subheader(f"Prediction: **{predicted_label}**")
                st.info(f"Confidence: **{confidence:.2f}%**")
                
                st.write("### All Class Probabilities:")
                for name, prob in zip(class_names, scores):
                    st.write(f"- **{name}**: {prob * 100:.2f}%")
        else:
            st.error("Model is not loaded properly. Cannot run prediction.")
