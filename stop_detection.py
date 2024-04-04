import cv2
import time
from ultralytics import YOLO  # Assuming you have YOLOv5 installed

# Initialize YOLOv8 model (you can replace this with your custom trained model)
model = YOLO('./detect/train5/weights/best.pt')

# Initialize OCR (you can use Tesseract or another OCR library)
def ocr_license_plate(image):
    # Implement OCR logic here
    # Return the license plate text
    pass

# Capture video from camera (or load video file)
cap = cv2.VideoCapture(1)  # Use camera index 0 (default webcam)

# video_path = 'test-stop-sign.mp4 '  # Specify the path to your video file
# cap = cv2.VideoCapture(video_path)
# print("success")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Detect license plates
    # results = model(frame, conf=.8)
    results = model.track(frame, conf=0.8, persist=True)
    
    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    print(boxes.cls.nelement() == 0)
    if boxes.cls.nelement():  # Assuming class 0 corresponds to license plates
        x1,y1,x2,y2 = boxes.xyxy[0]
        license_plate_image = frame[int(y1):int(y2), int(x1):int(x2)]
        # license_plate_text = ocr_license_plate(license_plate_image)

        # Save license plate image to a folder
        filename = f"license_plate_{time.time()}.jpg"
        cv2.imwrite(f"./license_plate_image/{filename}", license_plate_image)
        print(f"License plate saved: {filename}")

    # Display the frame (you can modify this part for visualization)
    cv2.imshow("License Plate Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()