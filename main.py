import customtkinter as ctk
import os
from model import Model
from view import View
from controller import Controller

# Debug için dosya yolunu kontrol et
print(f"Çalışma dizini: {os.getcwd()}")

def main():
    """Ana program fonksiyonu."""
    # Model oluştur
    model = Model(
        video_path="/Users/ardaaydin/Desktop/Skypulse/arabavid.mp4",
        model_path="best.pt"
    )
    
    # Kök pencereyi oluştur
    root = ctk.CTk()
    
    # View oluştur
    view = View(root)
    
    # Controller oluştur
    controller = Controller(model, view)
    
    # Uygulamayı başlat
    controller.run()


if __name__ == "__main__":
    main()