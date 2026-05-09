import streamlit as st
import cv2
import numpy as np
import os
import urllib.request

# -----------------------------
# Download Models Automatically
# -----------------------------
def download_models():
    os.makedirs("models", exist_ok=True)

    files = {
        "models/deploy.prototxt":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",

        "models/res10_300x300_ssd_iter_140000_fp16.caffemodel":
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel",

        "models/age_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",

        "models/age_net.caffemodel":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_net.caffemodel",

        "models/gender_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",

        "models/gender_net.caffemodel":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_net.caffemodel",
    }

    for file, url in files.items():
        if not os.path.exists(file):
            urllib.request.urlretrieve(url, file)

download_models()

# -----------------------------
# Load Models
# -----------------------------
face_net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

age_net = cv2.dnn.readNetFromCaffe(
    "models/age_deploy.prototxt",
    "models/age_net.caffemodel"
)

gender_net = cv2.dnn.readNetFromCaffe(
    "models/gender_deploy.prototxt",
    "models/gender_net.caffemodel"
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

            # Gender
            gender_net.setInput(blob_face)
            gender = GENDER_LIST[gender_net.forward()[0].argmax()]

            # Age
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
st.title("📸 Live Gender & Age Detection")

option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        result = predict(img)
        st.image(result, channels="BGR")

elif option == "Use Camera":
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        result = predict(img)
        st.image(result, channels="BGR")
