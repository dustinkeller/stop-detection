import datetime
import cv2
from ultralytics import YOLO  # Assuming you have YOLOv5 installed
from collections import defaultdict
from easyocr import Reader
import math
# import matplotlib.pyplot as plt

# Initialize YOLOv8 model (you can replace this with your custom trained model)
model = YOLO('./detect/train5/weights/best.pt')

# Initialize OCR
def ocr_license_plate(image, time):
    reader = Reader(['en'])
    cv2.threshold(image, 64, 225, cv2.THRESH_BINARY)
    plate_info = []
    result = reader.readtext(image)
    
    # Create a filename based on the capture time
    filename = f"ocr_results_{time}.txt"
    
    with open(f".\license_plate_ocr\{filename}", 'a') as file:
        for out in result:
            _, text, textScore = out
            if textScore > 0.4:
                plate_info.append(text)
                # Write the OCR result to the file
                file.write(f"Text: {text}, Score: {textScore}\n")

    return plate_info

# Once a plate has been detected, calculate the speed once tracking ends
def calculateSpeed(track, track_id, time):
    distance_pixels = math.sqrt((track[-1][0] - track[0][0])**2 + (track[-1][-1] - track[0][-1])**2)
    distance_meters = distance_pixels * pixel_to_meter_ratio
    time_seconds = len(track) / fps
    speed = (distance_meters / time_seconds) * 2.23694  # Convert m/s to mph
    
    # Label the speed
    if speed > mph_threshold:
        label = "Non-Stop"
    else:
        label = "Legal Stop"

    filename = f"ocr_results_{time}.txt"
    
    with open(f".\license_plate_ocr\{filename}", 'w') as file:
        file.write(f"Vehicle with ID {track_id} speed: {speed:.2f} mph - {label} \n")
    
    print(f"Vehicle with ID {track_id} speed: {speed:.2f} mph - {label}")

def calculate_pixel_to_meter_ratio(width, height):
    # Standard diagonal size of a display in inches
    diagonal_size_inches = 24

    # Convert diagonal size from inches to meters
    diagonal_size_meters = diagonal_size_inches * 0.0254

    # Calculate diagonal resolution using Pythagoras theorem
    diagonal_resolution = (width ** 2 + height ** 2) ** 0.5

    # Calculate pixels per meter ratio
    pixel_per_meter = diagonal_resolution / diagonal_size_meters

    return pixel_per_meter

# Capture video from camera (or load video file)
cap = cv2.VideoCapture(0)  # Use camera index 0 (default webcam)

# Store the track history
track_history = defaultdict(lambda: [])
captured_ids = set()  # Set to store IDs for which license plate images have been captured

# Parameters for calculating speed and camera specifications
fps = cap.get(cv2.CAP_PROP_FPS)
# Adjust this value based on your camera calibration and scene setup
pixel_to_meter_ratio = calculate_pixel_to_meter_ratio(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  
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
            print(f"Tracking: {track} from ID {track_ids}")
            if len(track) > 30:
                track.pop(0)

            # Check if the y value for track[0] is smaller than track[5], if so then it is an approaching vehicle
            if len(track) >= 6 and track[0][1] < track[5][1]:
                if track_ids not in captured_ids:  # Check if image for this ID has already been captured
                    newPlate = True
                    if boxes.cls.nelement():  # Assuming class 0 corresponds to license plates
                        x1, y1, x2, y2 = boxes.xyxy[0]
                        license_plate_image = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Save license plate image to a folder
                        cap_time = datetime.datetime.now()
                        filename = f"license_plate_{cap_time}.jpg"
                        cv2.imwrite(f".\license_plate_image\{filename}", license_plate_image)
                        captured_ids.add(track_ids)  # Add the ID to the set of captured IDs
                        if(newPlate):
                            calculateSpeed(track, track_ids,cap_time)
                            print(f"Frame: {len(track)}")

    # Display the frame (you can modify this part for visualization)
    cv2.imshow("License Plate Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
