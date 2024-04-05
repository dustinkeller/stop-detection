import cv2
import time
from ultralytics import YOLO  # Assuming you have YOLOv5 installed
from collections import defaultdict
import math

# Initialize YOLOv8 model (you can replace this with your custom trained model)
model = YOLO('./detect/train5/weights/best.pt')

# Initialize OCR (you can use Tesseract or another OCR library)
def ocr_license_plate(image):
    # Implement OCR logic here
    # Return the license plate text
    pass

# Once a plate has been detected, calculate the speed once tracking ends
def calculateSpeed(track, track_id):
    distance_pixels = math.sqrt((track[-1][0] - track[0][0])**2 + (track[-1][-1] - track[0][-1])**2)
    distance_meters = distance_pixels * pixel_to_meter_ratio
    time_seconds = len(track) / fps
    speed = (distance_meters / time_seconds) * 2.23694  # Convert m/s to mph
    
    # Label the speed
    if speed > mph_threshold:
        label = "Non-Stop"
    else:
        label = "Legal Stop"
    print(f"Vehicle with ID {track_id} speed: {speed:.2f} mph - {label}")
    
# Capture video from camera (or load video file)
cap = cv2.VideoCapture(0)  # Use camera index 0 (default webcam)

# Store the track history
track_history = defaultdict(lambda: [])
captured_ids = set()  # Set to store IDs for which license plate images have been captured

# Parameters for calculating speed
fps = cap.get(cv2.CAP_PROP_FPS)
pixel_to_meter_ratio = 0.02  # Adjust this value based on your camera calibration and scene setup
mph_threshold = 10  # Threshold speed in mph
annotated_frame = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Detect license plates
    results = model.track(frame, conf=0.8, persist=True, verbose=False)
    
    # Check if there are any detections
    if results[0].boxes:
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        try:
            track_ids = boxes.id.int().tolist()
        except AttributeError as err:
            continue

        for box, track_ids in zip(boxes, track_ids):
            newPlate = False
            x, y, w, h = box.xyxy[0]
            track = track_history[track_ids]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            # Check if the y value for track[0] is smaller than track[14], if so then it is an approaching vechile
            if len(track) >= 15 and track[0][1] < track[14][1]:
                if track_ids not in captured_ids:  # Check if image for this ID has already been captured
                    newPlate = True
                    if boxes.cls.nelement():  # Assuming class 0 corresponds to license plates
                        x1, y1, x2, y2 = boxes.xyxy[0]
                        license_plate_image = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Save license plate image to a folder
                        filename = f"license_plate_{time.time()}.jpg"
                        cv2.imwrite(f"./license_plate_image/{filename}", license_plate_image)
                        captured_ids.add(track_ids)  # Add the ID to the set of captured IDs
                        if(newPlate):
                            calculateSpeed(track, track_ids)

    # Display the frame (you can modify this part for visualization)
    cv2.imshow("License Plate Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
