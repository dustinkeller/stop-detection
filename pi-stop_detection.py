import cv2
import time
from ultralytics import YOLO
from collections import defaultdict
from easyocr import Reader
from picamera2 import Picamera2
import math

import json
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize YOLOv8 model
model = YOLO('./detect/train5/weights/best.pt')

# Initialize OCR
def ocr_license_plate(image, cap_time):
    reader = Reader(['en'])
    cv2.threshold(image, 64, 225, cv2.THRESH_BINARY)
    plate_info = []
    result = reader.readtext(image)
    
    # Create a filename based on the capture time
    filename = f"ocr_results_{cap_time}.txt"
    
    with open(filename, 'w') as file:
        for out in result:
            _, text, textScore = out
            if textScore > 0.4:
                plate_info.append(text)
                # Write the OCR result to the file
                file.write(f"Text: {text}, Score: {textScore}\n")

    cv2.imwrite(f"./license_plate_ocr/{filename}")
    return plate_info

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
    
# Store the track history
track_history = defaultdict(lambda: [])
captured_ids = set()  # Set to store IDs for which license plate images have been captured

# Parameters for calculating speed
fps = 30.0
pixel_to_meter_ratio = 0.02  # Adjust this value based on your camera calibration and scene setup
mph_threshold = 10  # Threshold speed in mph
annotated_frame = 0

# Initialize the camera
camera = Picamera2()
camera.video_configuration.controls.FrameRate = 30.0
# camera.preview_configuration.main.size=(700,380)
camera.preview_configuration.main.format="RGB888"
# camera.preview_configuration.align()
camera.configure("preview")
camera.start()

# Loop to capture frames
while True:
    # Capture frame from the camera
    frame = camera.capture_array()

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
            if len(track) > 10:
                track.pop(0)

            # Check if the y value for track[0] is smaller than track[2], if so then it is an approaching vehicle
            if len(track) >= 3 and track[0][1] < track[2][1]:
                if track_ids not in captured_ids:  # Check if image for this ID has already been captured
                    newPlate = True
                    if boxes.cls.nelement():  # Assuming class 0 corresponds to license plates
                        x1, y1, x2, y2 = boxes.xyxy[0]
                        license_plate_image = frame[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Save license plate image to a folder
                        cap_time = time.time()
                        filename = f"license_plate_{cap_time}.jpg"
                        cv2.imwrite(f"./license_plate_image/{filename}", license_plate_image)
                        print(ocr_license_plate(cv2.cvtColor(license_plate_image, cv2.COLOR_BGR2GRAY), cap_time))
                        captured_ids.add(track_ids)  # Add the ID to the set of captured IDs
                        if(newPlate):
                            calculateSpeed(track, track_ids)
                            print(f"Frame: {len(track)}")

    # Display the frame (you can modify this part for visualization)
    # cv2.imshow("License Plate Detection", annotated_frame)
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    if input("Press q to quit: ").lower()== 'q':
        break

# cv2.destroyAllWindows()

# Your Google Drive credentials file (service account key)
SERVICE_ACCOUNT_FILE = API

# Function to upload files to Google Drive folder
def upload_to_drive(files, folder_id):
    credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/drive'])
    drive_service = build('drive', 'v3', credentials=credentials)
    
    for file in files:
        file_metadata = {'name': file, 'parents': [folder_id]}
        media = MediaFileUpload(file, resumable=True)
        drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

# Get a list of image files in the license_plate_images folder
image_files = [f for f in os.listdir("license_plate_images") if f.endswith(".jpg")]

# Iterate through the image files and check if there is a corresponding OCR text file
for image_file in image_files:
    timestamp = image_file.split("_")[2].split(".")[0]
    ocr_file = f"ocr_results_{timestamp}.txt"
    if os.path.exists(os.path.join("license_plate_ocr", ocr_file)):
        files_to_upload = [os.path.join("license_plate_images", image_file), os.path.join("license_plate_ocr", ocr_file)]
        # Upload the files to Google Drive
        upload_to_drive(files_to_upload, 'your_folder_id')