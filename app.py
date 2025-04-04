import cv2
import base64
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from ultralytics import YOLO
import eventlet

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("yolov8s.pt")

# Open three video sources (Laptop webcam, Mobile camera 1, Mobile camera 2)
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture("http://192.168.137.62:8080/video")
cap3 = cv2.VideoCapture("http://172.16.7.153:8080/video")  # Make sure URL is correct (check `:` placement)

def generate_frames(camera_id, cap):
    """ Function to generate frames for each camera """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results[0].boxes.data
        people_count = 0

        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) == 0:
                people_count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"Person {people_count}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"People Count: {people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Emit both frame and count
        socketio.emit(f'video_frame_{camera_id}', {
            'image': frame_b64,
            'peopleCount': people_count
        })
        eventlet.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def connect():
    print("Client connected")
    socketio.start_background_task(generate_frames, "camera1", cap1)
    socketio.start_background_task(generate_frames, "camera2", cap2)
    socketio.start_background_task(generate_frames, "camera3", cap3)  # ← Add this line

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
