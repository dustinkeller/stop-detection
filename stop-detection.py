import datetime
import cv2
from ultralytics import YOLO  # Assuming you have YOLOv5 installed
from collections import defaultdict
from easyocr import Reader
import math
import os
# import matplotlib.pyplot as plt

# Initialize YOLOv8 model
model = YOLO('./detect/train5/weights/best.pt')

def ocr_license_plate(image, time):
    reader = Reader(['en'])
    cv2.threshold(image, 64, 225, cv2.THRESH_BINARY)
    plate_info = []
    result = reader.readtext(image)
    
    # Create a filename based on the capture time
    filename = f"ocr_results_{time}.txt"
    
    with open(f"./license_plate_ocr/{filename}", 'a') as file:
        for out in result:
            _, text, textScore = out
            if textScore > 0.2:
                plate_info.append(text)
                file.write(f"Text: {text}, Score: {textScore}\n")

    return plate_info

# Once a plate has been detected, calculate the speed once tracking ends
def calculateSpeed(track, track_id, time):
    distance_pixels = math.sqrt((track[-1][0] - track[0][0])**2 + (track[-1][-1] - track[0][-1])**2)
    distance_meters = distance_pixels * pixel_to_meter_ratio
    time_seconds = len(track) / fps
    speed = (distance_meters / time_seconds) * 2.23694  # Convert m/s to mph
    
    if speed > mph_threshold:
        label = "Non-Stop"
    else:
        label = "Legal Stop"

    filename = f"ocr_results_{time}.txt"
    
    with open(f"./license_plate_ocr/{filename}", 'w') as file:
        file.write(f"Vehicle with ID {track_id} speed: {speed:.2f} mph - {label} \n")
    
    print(f"Vehicle with ID {track_id} speed: {speed:.2f} mph - {label}")

def calculate_pixel_to_meter_ratio(frame_width, frame_height):
    # The physical width and height of the scene being captured (in meters)
    scene_width = 40.0
    scene_height = 40.0

    # Calculate the pixel to meter ratios for the width and height
    pixel_to_meter_ratio_width = scene_width / frame_width
    pixel_to_meter_ratio_height = scene_height / frame_height

    # Return the average of the two ratios
    return (pixel_to_meter_ratio_width + pixel_to_meter_ratio_height) / 2



# Capture video from camera
cap = cv2.VideoCapture(1)  # Camera index 0 for default webcam

# Capturing from a video file
# cap = cv2.VideoCapture("./test-stop-sign.mp4")

# Store the track history
track_history = defaultdict(lambda: [])
captured_ids = set()  # Set to store IDs for which license plate images have been captured

# Parameters for calculating speed and camera specifications
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
# Value based on camera calibration and scene setup
pixel_to_meter_ratio = calculate_pixel_to_meter_ratio(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(pixel_to_meter_ratio) 
mph_threshold = 20  # Threshold speed in mph
annotated_frame = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Run model inference on frame to track any license plates
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
            # print(f"Tracking: {track} from ID {track_ids}")
            if len(track) > 90:
                track.pop(0)

            # Check if the y value for the initial frame vs the last "preliminary" tracking frame
            if 80 <= len(track) <= 90 and track[0][1] < track[-1][1]:
                if track_ids not in captured_ids:  # Check if image for this ID has already been captured
                    newPlate = True
                    if boxes.cls.nelement():  # Assuming class 0 corresponds to license plates
                        x1, y1, x2, y2 = boxes.xyxy[0]
                        license_plate_image = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Save license plate image to a folder
                        cap_time = datetime.datetime.now()
                        filename = f"license_plate_{cap_time}.jpg"
                        cv2.imwrite(f"./license_plate_image/{filename}", license_plate_image)
                        captured_ids.add(track_ids)  # Add the ID to the set of captured IDs
                        if(newPlate):
                            calculateSpeed(track, track_ids,cap_time)
                            print(f"Frame: {len(track)}")

    cv2.imshow("License Plate Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Get a list of image files in the license_plate_images folder
image_files = [f for f in os.listdir("license_plate_image") if f.endswith(".jpg")]

# Iterate through the image files and check if there is a corresponding OCR text file
for image_file in image_files:
    timestamp = image_file.split("_")[2][:-4]
    true_img = cv2.imread(f"license_plate_image/{image_file}")
    ocr_license_plate(cv2.cvtColor(true_img, cv2.COLOR_BGR2GRAY),timestamp)
    ocr_file = f"ocr_results_{timestamp}.txt"

cap.release()
cv2.destroyAllWindows()
