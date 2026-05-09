import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("📸 Gender & Age Prediction (No OpenCV)")

# Load pre-trained model (dummy for now)
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

model = load_model()

# Camera input
camera_image = st.camera_input("Take a picture")

if camera_image:
    img = Image.open(camera_image)
    st.image(img, caption="Captured Image")

    # Preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Prediction (dummy classification)
    preds = model.predict(img_array)
    label = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0][1]

    st.success("Prediction Done ✅")

    # Replace with real model later
    st.write(f"Detected Object: {label}")
    st.write("Gender: Male")
    st.write("Age: (25-32)")
