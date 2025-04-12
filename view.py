import customtkinter as ctk
from PIL import Image, ImageTk
from typing import Dict, Any, Callable

class View:
    """
    Kullanıcı arayüzünü temsil eden sınıf.
    """
    def __init__(self, root: ctk.CTk) -> None:
        """
        View sınıfının yapıcı metodu.
        
        Args:
            root: CTk kök penceresi
        """
        self.root = root
        self.root.title("Sky Pulse Management")
        
        # Tema ayarları
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # GUI bileşenleri
        self.frames = {}
        self.canvas = None
        self.detected_objects_label = None
        self.values_label = None
        self.class_selection_menu = None
        self.attack_method_menu = None
        
        # GUI oluştur
        self._setup_grid()
        self._create_frames()
        self._setup_video_area()
        self._setup_info_area()
        self._setup_detected_objects_area()
        self._setup_control_area()
        
    def _setup_grid(self) -> None:
        """Ana pencerenin grid yapısını ayarlar."""
        self.root.grid_columnconfigure(0, weight=2, uniform="column")
        self.root.grid_columnconfigure(1, weight=1, uniform="column")
        self.root.grid_rowconfigure(0, weight=3, uniform="row")
        self.root.grid_rowconfigure(1, weight=1, uniform="row")
        
    def _create_frames(self) -> None:
        """Dört ana bölme çerçevesini oluşturur."""
        frame_properties = [
            ("video", 0, 0, "red"),      # Video görüntüleme alanı
            ("info", 0, 1, "blue"),      # Bilgi ekranı
            ("detection", 1, 0, "green"), # Tespit bilgileri
            ("control", 1, 1, "yellow")   # Kontrol butonları
        ]
        
        for name, row, col, color in frame_properties:
            frame = ctk.CTkFrame(self.root, corner_radius=10, border_width=2, border_color=color)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            self.frames[name] = frame
            
    def _setup_video_area(self) -> None:
        """Video görüntüleme alanını ayarlar."""
        self.canvas = ctk.CTkCanvas(self.frames["video"])
        self.canvas.pack(fill="both", expand=True)
        
    def _setup_info_area(self) -> None:
        """Bilgi ekranı alanını ayarlar."""
        try:
            # Logo
            logo_img = Image.open("main.png")
            logo_img = logo_img.resize((200, 200))
            logo_photo = ctk.CTkImage(light_image=logo_img, dark_image=logo_img, size=(200, 200))
            logo_label = ctk.CTkLabel(self.frames["info"], image=logo_photo, text="")
            logo_label.image = logo_photo
            logo_label.pack(pady=20)
            
            # Değerler için label
            self.values_label = ctk.CTkLabel(self.frames["info"], text="", text_color="white")
            self.values_label.pack(pady=10)
            
            # Seçenek menüsü
            options_label = ctk.CTkLabel(self.frames["info"], text="Saldırma Yöntemi", text_color="white")
            options_label.pack(pady=10)
            self.attack_method_menu = ctk.CTkOptionMenu(
                self.frames["info"], 
                values=["Düz Saldır", "Zikzak çiz", "Dönerek Saldır"]
            )
            self.attack_method_menu.pack(pady=5)
            
        except Exception as e:
            print(f"Logo yüklenirken hata oluştu: {e}")
            
    def _setup_detected_objects_area(self) -> None:
        """Tespit edilen nesneler alanını ayarlar."""
        self.detected_objects_label = ctk.CTkLabel(self.frames["detection"], text="", text_color="white")
        self.detected_objects_label.pack(pady=20)
        
    def _setup_control_area(self) -> None:
        """Kontrol butonları alanını ayarlar."""
        # Butonlar setup_control_buttons metodu içinde oluşturulacak
        pass
        
    def setup_control_buttons(self, commands: Dict[str, Callable]) -> None:
        """
        Kontrol butonlarını oluşturur ve komutlarını bağlar.
        
        Args:
            commands: Buton komutlarını içeren sözlük
        """
        button_configs = [
            ("Yaklaş", commands.get("zoom"), "#006400", "green", 0, 0),
            ("Farklı Açı Bak", commands.get("angle"), "#FF8C00", "orange", 1, 0),
            ("Saldır", commands.get("attack"), "#8B0000", "red", 2, 0),
            ("Çıkış", commands.get("exit"), "#1C1C1C", "gray", 3, 0)
        ]
        
        for text, command, fg_color, hover_color, row, col in button_configs:
            button = ctk.CTkButton(
                self.frames["control"], 
                text=text,
                command=command,
                fg_color=fg_color, 
                hover_color=hover_color, 
                text_color="white"
            )
            button.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
        # Sınıf seçim menüsü
        self.class_selection_menu = ctk.CTkOptionMenu(
            self.frames["control"], 
            values=["Default", "Person", "Car"],
            command=commands.get("select_class")
        )
        self.class_selection_menu.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
    def update_video_display(self, frame: Any) -> None:
        """
        Video görüntüsünü günceller.
        
        Args:
            frame: Görüntülenecek video karesi
        """
        if frame is not None:
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=photo, anchor="nw")
            self.canvas.image = photo
            
    def update_detected_objects(self, objects_dict: Dict[str, list], selected_class: str) -> None:
        """
        Tespit edilen nesnelerin bilgilerini günceller.
        
        Args:
            objects_dict: Tespit edilen nesneler sözlüğü
            selected_class: Seçilen nesne sınıfı
        """
        label_text = ""
        objects_list = objects_dict.get(selected_class, [])
        
        for idx, (obj, confidence, track_id) in enumerate(objects_list, start=1):
            label_text += f"ID:{int(track_id)} {selected_class.capitalize()}{idx}: Confidence {confidence:.2f}\n"
            
        self.detected_objects_label.configure(text=label_text)
        
    def update_sensor_values(self, values: Dict[str, float]) -> None:
        """
        Sensör değerlerini günceller.
        
        Args:
            values: Güncellenecek sensör değerleri
        """
        values_text = f"Value1: {values['value1']}\nValue2: {values['value2']}\nValue3: {values['value3']}"
        self.values_label.configure(text=values_text) 