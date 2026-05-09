import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# --------------------------
# Load Pretrained Model
# --------------------------
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # dummy output
model.eval()

# --------------------------
# Transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------------
# Fake Labels (for demo)
# --------------------------
GENDER = ["Male", "Female"]
AGE = ["(0-10)", "(10-20)", "(20-30)", "(30-40)",
       "(40-50)", "(50-60)", "(60-70)", "(70-80)"]

# --------------------------
# Predict Function
# --------------------------
def predict(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    # Dummy logic (replace with real trained model if needed)
    gender = GENDER[torch.randint(0, 2, (1,)).item()]
    age = AGE[torch.randint(0, 8, (1,)).item()]

    return gender, age

# --------------------------
# UI
# --------------------------
st.title("📸 Age & Gender Detection (PyTorch)")

option = st.radio("Choose Input", ["Camera", "Upload Image"])

# 📷 Camera
if option == "Camera":
    img_file = st.camera_input("Take Photo")

    if img_file:
        image = Image.open(img_file)
        gender, age = predict(image)

        st.image(image)
        st.success(f"Prediction: {gender}, {age}")

# 📁 Upload
else:
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if img_file:
        image = Image.open(img_file)
        gender, age = predict(image)

        st.image(image)
        st.success(f"Prediction: {gender}, {age}")
        
