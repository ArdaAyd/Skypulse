import cv2
from typing import Dict, Callable
from model import Model
from view import View

class Controller:
    """
    Model ve View arasındaki iletişimi yöneten sınıf.
    """
    def __init__(self, model: Model, view: View) -> None:
        """
        Controller sınıfının yapıcı metodu.
        
        Args:
            model: Uygulama modeli
            view: Kullanıcı arayüzü
        """
        self.model = model
        self.view = view
        
        # Kontrol komutlarını tanımla
        self.commands = {
            "zoom": lambda: self._on_button_click(1),
            "angle": lambda: self._on_button_click(2),
            "attack": lambda: self._on_button_click(3),
            "exit": self._close_program,
            "select_class": self._on_class_selected
        }
        
        # Butonları ayarla
        self.view.setup_control_buttons(self.commands)
        
        # Zamanlayıcıları başlat
        self._start_timers()
        
    def _start_timers(self) -> None:
        """Periyodik görevleri başlatır."""
        self._update_video()
        self._update_sensor_values()
        
    def _update_video(self) -> None:
        """Video görüntüsünü ve nesne tespitlerini günceller."""
        frame, objects = self.model.process_video_frame()
        
        if frame is not None:
            self.view.update_video_display(frame)
            self.view.update_detected_objects(objects, self.model.get_selected_class())
            
        self.view.canvas.after(1, self._update_video)
        
    def _update_sensor_values(self) -> None:
        """Sensör değerlerini günceller."""
        values = self.model.get_sensor_data()
        self.view.update_sensor_values(values)
        self.view.values_label.after(500, self._update_sensor_values)
        
    def _on_button_click(self, button_id: int) -> None:
        """
        Buton tıklama olayını işler.
        
        Args:
            button_id: Tıklanan butonun ID'si
        """
        self.view.detected_objects_label.configure(text=f"Frame {button_id} butonuna tıklandı!")
        
    def _on_class_selected(self, selected: str) -> None:
        """
        Sınıf seçildiğinde çağrılır.
        
        Args:
            selected: Seçilen sınıf
        """
        current_class = self.model.get_selected_class()
        if current_class != selected:
            self.model.set_selected_class(selected)
            print(f"Sınıf değiştirildi: {selected}")
            
    def _close_program(self) -> None:
        """Programı kapatır."""
        self.model.release_resources()
        cv2.destroyAllWindows()
        self.view.root.destroy()
        
    def run(self) -> None:
        """Ana döngüyü başlatır."""
        self.view.root.mainloop() 