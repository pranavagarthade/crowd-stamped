import cv2
import base64
import numpy as np
from flask import Flask, render_template, send_file
from flask_socketio import SocketIO
from ultralytics import YOLO
import eventlet
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("yolov8s.pt")

# Video Sources
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture("http://192.168.137.62:8080/video") #Make use of IP webcam app from Playstore
cap3 = cv2.VideoCapture("http://172.16.7.153:8080/video") #Make use of IP webcam app from Playstore

# Initialize peak counters
peak_counts = {
    "camera1": 0,
    "camera2": 0,
    "camera3": 0
}

def save_peak_counts():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Define column names
    columns = ["timestamp", "exit1", "exit2", "exit3"]

    # Create DataFrame with one row and proper columns
    row = pd.DataFrame([[timestamp, peak_counts["camera1"], peak_counts["camera2"], peak_counts["camera3"]]], columns=columns)

    csv_path = "C:/Users/acer/Desktop/temp/crowd_data.csv"

    # Check if file exists
    if os.path.exists(csv_path):
        row.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        row.to_csv(csv_path, mode='w', header=True, index=False)

def generate_frames(camera_id, cap):
    global peak_counts
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

        # Update peak
        if people_count > peak_counts[camera_id]:
            peak_counts[camera_id] = people_count

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        socketio.emit(f'video_frame_{camera_id}', {
            'image': frame_b64,
            'peopleCount': people_count
        })

        eventlet.sleep(0.05)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/trend-graph')
# def trend_graph():
#     print("Trend graph endpoint hit")
#     try:
#         df = pd.read_csv(data_file)
#         print("CSV Loaded:")
#         print(df.head())

#         if df.empty:
#             print("CSV is empty!")
#             return "No data available", 500

#         df['timestamp'] = pd.to_datetime(df['timestamp'])

#         plt.figure(figsize=(10, 6))
#         plt.plot(df['timestamp'], df['exit1'], label='Exit 1')
#         plt.plot(df['timestamp'], df['exit2'], label='Exit 2')
#         plt.plot(df['timestamp'], df['exit3'], label='Exit 3')
#         plt.xlabel('Time')
#         plt.ylabel('People Count')
#         plt.title('Crowd Count Trend Over Time')
#         plt.legend()
#         plt.tight_layout()

#         img = io.BytesIO()
#         plt.savefig(img, format='png')
#         plt.close()
#         img.seek(0)
#         print("Graph generated successfully")
#         return send_file(img, mimetype='image/png')

#     except Exception as e:
#         print("Error generating trend graph:", e)
#         return "Failed to generate graph", 500



@socketio.on('connect')
def connect():
    print("Client connected")
    socketio.start_background_task(generate_frames, "camera1", cap1)
    socketio.start_background_task(generate_frames, "camera2", cap2)
    socketio.start_background_task(generate_frames, "camera3", cap3)

    eventlet.spawn_after(30, save_peak_counts)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
