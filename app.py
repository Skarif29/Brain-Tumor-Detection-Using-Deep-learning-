import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.title("Brain Tumor Detection App")

# Dataset-er alphabetical class labels
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

@st.cache_resource
def load_model_dynamically():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(current_dir):
        if file.startswith("brain_tumor_model") or file.endswith(".h5"):
            full_path = os.path.join(current_dir, file)
            return tf.keras.models.load_model(full_path)
    raise FileNotFoundError("No .h5 model file found.")

try:
    model = load_model_dynamically()
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Upload an MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Image load & RGB Mode enforce
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # 2. Image Resize (128x128)
    img = image.resize((128, 128))
    
    # 3. Preprocessing array
    img_array = np.array(img, dtype=np.float32)
    
    # Range Normalization (Jodi 0-1 scale target thake)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            prediction = model.predict(img_array)
            
            # Predict scores calculate
            scores = prediction[0]
            predicted_class_idx = np.argmax(scores)
            predicted_label = class_names[predicted_class_idx]
            confidence = float(scores[predicted_class_idx]) * 100
            
            st.write("---")
            st.subheader(f"Prediction: **{predicted_label}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
            
            # Sub-level class breakdown table
            st.write("### All Class Probabilities:")
            for name, prob in zip(class_names, scores):
                st.write(f"- **{name}**: {prob * 100:.2f}%")