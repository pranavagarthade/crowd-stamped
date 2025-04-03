import cv2 
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(0)

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
            cv2.rectangle(frame, (int(x1), int(y1), int(x2), int(y2)), (0, 255, 0), 2)

            label = f"Person {people_count}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"People Count: {people_count}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if people_count > 2:
        cv2.putText(frame, f"Crowd Detected", (400,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Live Crowd Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()