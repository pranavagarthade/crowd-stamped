import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

cap1 = cv2.VideoCapture(0)  # Laptop webcam
cap2 = cv2.VideoCapture("http://192.168.137.62:4747/video")  # Mobile camera (IP stream)

if not cap1.isOpened():
    print("Error: Could not open webcam.")
if not cap2.isOpened():
    print("Error: Could not open phone camera.")

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Could not read frame from one or both cameras.")
        break

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
            label = f"Person {camera1_people_count}"
            cv2.putText(frame1, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame1, f"People Count: {camera1_people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if camera1_people_count > 2:
        cv2.putText(frame1, "Crowd Detected!", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for det in detections2:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) == 0:
            camera2_people_count += 1
            cv2.rectangle(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Person {camera2_people_count}"
            cv2.putText(frame2, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame2, f"People Count: {camera2_people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if camera2_people_count > 2:
        cv2.putText(frame2, "Crowd Detected!", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    combined = cv2.hconcat([frame1, frame2])
    cv2.imshow("Crowd Detection (Laptop & Phone)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
