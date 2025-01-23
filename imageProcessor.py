from ultralytics import YOLO
import cv2
import time

class ImageProcessor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
    def process_frame(self, frame, selected_class):
        self.new_frame_time = time.time()
        fps = 1/(self.new_frame_time-self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.new_frame_time
        fps = int(fps)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if selected_class == "Default":
            class_indices = None
        elif selected_class == "Person":
            class_indices = [0]
        elif selected_class == "Car":
            class_indices = [2, 3, 5, 7]
        else:
            class_indices = None

        results = self.model.track(rgb_frame, 
                                 classes=class_indices, 
                                 persist=False,
                                 conf=0.25,
                                 iou=0.45,
                                 imgsz=640)

        output_frame = results[0].plot()
        
        cv2.putText(output_frame, f"FPS: {fps}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

        classified_objects = {}
        for result in results[0].boxes:
            cls = result.cls.item()
            track_id = result.id.item() if result.id is not None else -1
            
            if selected_class == "Default" or cls in class_indices:
                if selected_class not in classified_objects:
                    classified_objects[selected_class] = []
                classified_objects[selected_class].append((result.xyxy.numpy(), result.conf.item(), track_id))

        return output_frame, classified_objects

