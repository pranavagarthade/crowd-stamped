import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

cap1 = cv2.VideoCapture(0)  # Laptop webcam
cap2 = cv2.VideoCapture("http://192.168.137.62:8080/video")  # Mobile camera (IP stream)

if not cap1.isOpened():
    print("Error: Could not open webcam.")
if not cap2.isOpened():
    print("Error: Could not open phone camera.")

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or frame1 is None:
        print("Error: Could not read frame from Laptop Webcam.")
        continue
    if not ret2 or frame2 is None:
        print("Error: Could not read frame from Mobile Camera.")
        continue

    # Resize phone camera frame to match laptop webcam size
    frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    results1 = model(frame1)
    results2 = model(frame2)

    detections1 = results1[0].boxes.data
    detections2 = results2[0].boxes.data

    camera1_people_count = 0
    camera2_people_count = 0

    for det in detections1:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) == 0: 
            camera1_people_count += 1
            cv2.rectangle(frame1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame1, f"Person {camera1_people_count}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame1, f"People Count: {camera1_people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for det in detections2:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) == 0:
            camera2_people_count += 1
            cv2.rectangle(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame2, f"Person {camera2_people_count}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame2, f"People Count: {camera2_people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show each frame in a separate window
    cv2.imshow("Crowd Detection - Laptop Webcam", frame1)
    cv2.imshow("Crowd Detection - Mobile Camera", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
