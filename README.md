# SensorStop
SensorStop is a stop sign speed monitoring system, encouraging drivers to safely make complete stops at stop signs and follow the speed limit.

## Usage
Install all [dependencies](#Dependencies). Also ensure that the model `best.pt` is in `./detect/train5/weights/`.

From the terminal, navigate to the project folder and run:
```
python3 stop-detection.py
```
(If you are using this on a Raspberry Pi 4B with an Arducam IMX708, replace `stop-detection.py` with `pi-stop-detection.py`)

Make sure the camera connected to your computer is facing traffic from the perspective of a stop sign. From there, the system can detect license plates as well as full stops, and calculate speed data which is all saved to the computer in the `./license_plate_ocr/` and `./license_plate_image/` directories.

## Development and Implementation Information
Want to learn more about the development of SensorStop, its purpose, and its implementation, check out our [wiki](https://github.com/dustinkeller/stop-detection/wiki)!

## Other Relevant Repositories
- [Web Dashboard](https://github.com/syedalif/sensor-stop-dashboard)
- [Object Detection Model Training](https://github.com/dustinkeller/yolov8-plate-detection)
- [License Plate Data Preprocessing](https://github.com/dustinkeller/license-data-preprocessing)

## Dependencies
- OpenCV
- Ultralytics
- EasyOCR
- Picamera2
