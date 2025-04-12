import cv2
import sys
from typing import Dict, Any, Tuple
from imageProcessor import ImageProcessor
from data import get_numeric_data1, get_numeric_data2, get_numeric_data3

class Model:
    """
    Uygulamanın veri modelini temsil eden sınıf.
    """
    def __init__(self, video_path: str, model_path: str) -> None:
        """
        Model sınıfının yapıcı metodu.
        
        Args:
            video_path: Video dosyasının yolu
            model_path: YOLO modelinin yolu
        """
        self.video_path = video_path
        self.cap = self._init_video_capture()
        self.image_processor = ImageProcessor(model_path)
        self.selected_class = "Default"
        self.detected_objects = {}
        self.frame_count = 0
        self.frame_skip = 2  # Her kaç karede bir işlem yapılacağı
        
    def _init_video_capture(self) -> cv2.VideoCapture:
        """Video yakalama nesnesini başlatır."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Video açılamadı! Programdan çıkılıyor...")
            sys.exit()
        return cap
        
    def set_selected_class(self, new_class: str) -> None:
        """Seçilen sınıfı günceller."""
        self.selected_class = new_class
        
    def get_selected_class(self) -> str:
        """Seçilen sınıfı döndürür."""
        return self.selected_class
        
    def process_video_frame(self) -> Tuple[Any, Dict]:
        """Video karesini işler ve sonuçları döndürür."""
        self.frame_count += 1
        
        # Her N karede bir işlem yap
        if self.frame_count % self.frame_skip != 0:
            return None, {}
            
        ret, frame = self.cap.read()
        if not ret:
            print("Video frame'i okunamadı")
            return None, {}
            
        self.detected_objects.clear()
        processed_frame, classified_objects = self.image_processor.process_frame(
            frame, 
            self.selected_class
        )
        
        return processed_frame, classified_objects
        
    def get_sensor_data(self) -> Dict[str, float]:
        """Sensör verilerini alır."""
        return {
            "value1": get_numeric_data1(),
            "value2": get_numeric_data2(),
            "value3": get_numeric_data3()
        }
        
    def release_resources(self) -> None:
        """Kaynakları serbest bırakır."""
        self.cap.release() 