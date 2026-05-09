import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os

# --------------------------
# Download model if not exists
# --------------------------
MODEL_PATH = "model.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1q9q5xZz_dummy_model_link"  # replace if needed
    gdown.download(url, MODEL_PATH, quiet=False)

# --------------------------
# Load Model
# --------------------------
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

# --------------------------
# Transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------------
# Labels
# --------------------------
GENDER = ['Male', 'Female']
AGE = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70+']

# --------------------------
# Predict
# --------------------------
def predict(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    gender = GENDER[output[0][0].argmax().item()]
    age = AGE[output[1][0].argmax().item()]

    return gender, age

# --------------------------
# UI
# --------------------------
st.title("📸 Age & Gender Detection (Real Model)")

option = st.radio("Choose Input", ["Camera", "Upload"])

if option == "Camera":
    img_file = st.camera_input("Take Photo")
else:
    img_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if img_file:
    image = Image.open(img_file)
    gender, age = predict(image)

    st.image(image)
    st.success(f"Prediction: {gender}, {age}")
