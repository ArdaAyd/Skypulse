from ultralytics import YOLO
import cv2
import time
from typing import Dict, List, Tuple, Optional, Union, Any

class ImageProcessor:
    """
    Video kareleri üzerinde YOLO modeli kullanarak nesne tespiti yapan sınıf.
    """
    
    # Sınıf sabitleri
    CLASS_INDICES = {
        "Default": None,
        "Person": [0],
        "Car": [2, 3, 5, 7]
    }
    
    def __init__(self, model_path: str) -> None:
        """
        ImageProcessor sınıfının yapıcı metodu.
        
        Args:
            model_path: YOLO modelinin dosya yolu
        """
        self.model = YOLO(model_path)
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.tracked_objects = {}  # Takip edilen nesneleri saklamak için
    
    def _calculate_fps(self) -> int:
        """
        FPS (Frames Per Second) hesaplar.
        
        Returns:
            Saniyedeki kare sayısı
        """
        self.new_frame_time = time.time()
        fps = 1/(self.new_frame_time-self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.new_frame_time
        return int(fps)
    
    def _get_class_indices(self, selected_class: str) -> Optional[List[int]]:
        """
        Seçilen sınıf için YOLO indekslerini döndürür.
        
        Args:
            selected_class: Seçilen nesne sınıfı
            
        Returns:
            Sınıf indeksleri listesi veya None
        """
        return self.CLASS_INDICES.get(selected_class, None)
    
    def process_frame(self, frame: Any, selected_class: str) -> Tuple[Any, Dict[str, List]]:
        """
        Video karesini işler ve tespit edilen nesneleri döndürür.
        
        Args:
            frame: İşlenecek video karesi
            selected_class: Tespit edilecek nesne sınıfı
            
        Returns:
            İşlenmiş kare ve tespit edilen nesneler sözlüğü
        """
        fps = self._calculate_fps()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        class_indices = self._get_class_indices(selected_class)
        
        results = self.model.track(
            rgb_frame, 
            classes=class_indices, 
            persist=True,
            conf=0.25,
            iou=0.45,
            imgsz=640
        )
        
        output_frame = results[0].plot()
        
        # FPS metnini ekle
        cv2.putText(
            output_frame, 
            f"FPS: {fps}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (100, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        
        classified_objects = self._extract_detected_objects(results, selected_class, class_indices)
        return output_frame, classified_objects
    
    def _extract_detected_objects(self, results: Any, selected_class: str, class_indices: Optional[List[int]]) -> Dict[str, List]:
        """
        YOLO sonuçlarından nesne bilgilerini çıkarır.
        
        Args:
            results: YOLO model sonuçları
            selected_class: Seçilen nesne sınıfı
            class_indices: Sınıf indeksleri
            
        Returns:
            Sınıflandırılmış nesneler sözlüğü
        """
        classified_objects = {}
        
        for result in results[0].boxes:
            cls = result.cls.item()
            track_id = result.id.item() if result.id is not None else -1
            bbox = result.xyxy.numpy()
            conf = result.conf.item()
            
            if selected_class == "Default" or cls in class_indices:
                if selected_class not in classified_objects:
                    classified_objects[selected_class] = []
                classified_objects[selected_class].append((bbox, conf, track_id))
                
                # Takip edilen nesneleri kaydet
                self.tracked_objects[track_id] = bbox
                
        return classified_objects
