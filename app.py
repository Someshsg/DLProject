import streamlit as st
import numpy as np
import cv2
import os
import urllib.request

# Fix OpenCV issue
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# -----------------------------
# Create models folder if not exists
# -----------------------------
if not os.path.exists("models"):
    os.makedirs("models")

# -----------------------------
# Download models automatically
# -----------------------------
def download_model(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

# Face model
download_model(
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "models/deploy.prototxt"
)
download_model(
    "https://github.com/opencv/opencv_3rdparty/raw/master/dnn_models/res10_300x300_ssd_iter_140000.caffemodel",
    "models/res10.caffemodel"
)

# Age model
download_model(
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    "models/age.prototxt"
)
download_model(
    "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
    "models/age.caffemodel"
)

# Gender model
download_model(
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "models/gender.prototxt"
)
download_model(
    "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
    "models/gender.caffemodel"
)

# -----------------------------
# Load models
# -----------------------------
face_net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10.caffemodel"
)

age_net = cv2.dnn.readNetFromCaffe(
    "models/age.prototxt",
    "models/age.caffemodel"
)

gender_net = cv2.dnn.readNetFromCaffe(
    "models/gender.prototxt",
    "models/gender.caffemodel"
)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

GENDER_LIST = ['Male', 'Female']

# -----------------------------
# Prediction Function
# -----------------------------
def predict(img):
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face = img[y1:y2, x1:x2]

            if face.size == 0:
                continue

            blob_face = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.426, 87.768, 114.895), swapRB=False
            )

            gender_net.setInput(blob_face)
            gender = GENDER_LIST[gender_net.forward()[0].argmax()]

            age_net.setInput(blob_face)
            age = AGE_LIST[age_net.forward()[0].argmax()]

            label = f"{gender}, {age}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img

# -----------------------------
# UI
# -----------------------------
st.title("📸 Gender & Age Detection App")

option = st.radio("Choose Input:", ["Upload Image", "Use Camera"])

# Upload
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        result = predict(img)
        st.image(result, channels="BGR")

# Camera
elif option == "Use Camera":
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        result = predict(img)
        st.image(result, channels="BGR")
