# imageProcessor.py

from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

def process_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    
    # YOLO inference
    results = model(frame)
    
    # Render the results to the frame
    output_frame = results[0].plot()  # plot() ile çizim yapıyoruz
    
    return output_frame
