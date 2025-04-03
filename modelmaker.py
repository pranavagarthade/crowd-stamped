import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam and phone camera
cap1 = cv2.VideoCapture(0)  # Laptop webcam
cap2 = cv2.VideoCapture("http://192.168.29.134:4747/video")  # Mobile camera (IP stream)

# Check if cameras opened successfully
if not cap1.isOpened():
    print("Error: Could not open webcam.")
if not cap2.isOpened():
    print("Error: Could not open phone camera.")

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Exit loop if either frame isn't available
    if not ret1 or not ret2:
        print("Error: Could not read frame from one or both cameras.")
        break

    # Run YOLO detection on both frames
    results1 = model(frame1)
    results2 = model(frame2)

    # Extract detected objects
    detections1 = results1[0].boxes.data
    detections2 = results2[0].boxes.data

    # People count for both cameras
    camera1_people_count = 0
    camera2_people_count = 0

    # Process detections for Laptop Webcam
    for det in detections1:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) == 0:  # Class 0 = person
            camera1_people_count += 1
            cv2.rectangle(frame1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Person {camera1_people_count}"
            cv2.putText(frame1, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame1, f"People Count: {camera1_people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if camera1_people_count > 2:
        cv2.putText(frame1, "Crowd Detected!", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Process detections for Phone Camera
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

    # Display both cameras side by side
    combined = cv2.hconcat([frame1, frame2])
    cv2.imshow("Crowd Detection (Laptop & Phone)", combined)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
