
from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt","res10_300x300_ssd_iter_140000.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt","age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt","gender_net.caffemodel")

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)','(38-43)','(48-53)','(60-100)']
gender_list = ['Male','Female']
MODEL_MEAN_VALUES = (78.426,87.768,114.895)

cap = cv2.VideoCapture(0)

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h,w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),(104,117,123))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf > 0.6:
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                x1,y1,x2,y2 = box.astype(int)

                face = frame[y1:y2,x1:x2]
                if face.size == 0:
                    continue

                blob2 = cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)

                gender_net.setInput(blob2)
                gender = gender_list[gender_net.forward()[0].argmax()]

                age_net.setInput(blob2)
                age = age_list[age_net.forward()[0].argmax()]

                label = f"{gender}, {age}"

                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        _,buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
