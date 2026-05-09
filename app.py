import streamlit as st
import numpy as np
from PIL import Image

st.title("📸 Gender & Age Detection")

st.write("Choose input method:")

# Option selection
option = st.radio("Select Option", ["Use Camera", "Upload Image"])

# -----------------------------
# CAMERA INPUT
# -----------------------------
if option == "Use Camera":
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        img = Image.open(camera_image)
        st.image(img, caption="Captured Image")

        st.success("Image Captured Successfully ✅")

        # Dummy prediction
        st.write("Gender: Male")
        st.write("Age: (25-32)")

# -----------------------------
# FILE UPLOAD
# -----------------------------
elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image")

        st.success("Image Uploaded Successfully ✅")

        # Dummy prediction
        st.write("Gender: Female")
        st.write("Age: (20-25)")
