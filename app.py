import streamlit as st
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# === Step 1: Custom CSS for Styling ===
st.markdown("""
    <style>
    body {
        background-color: #f4f6f8;
    }
    .main {
        background-color: white;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #20639B;
        text-align: center;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 24px;
    }
    </style>
""", unsafe_allow_html=True)

# === Step 2: Load Model from Google Drive ===
model_filename = "retina_model.h5"
file_id = "1FsEbL0MZcoSBgIFkNcNnQSmElB2w4OyU"  # <-- your model's file ID

if not os.path.exists(model_filename):
    st.text("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_filename, quiet=False)

model = load_model(model_filename)

# === Step 3: Class Labels (edit if different) ===
class_names = [
    'Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Age-related Macular Degeneration (AMD)', 'Other'
]

# === Step 4: App Title ===
st.title("👁️ Retinal Disorder Classifier")
st.subheader("Upload a fundus image to predict the condition")

# === Step 5: Upload Image ===
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🩺 Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
