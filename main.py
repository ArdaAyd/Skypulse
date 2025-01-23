import customtkinter as ctk
import cv2
import sys
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import argparse
from data import get_numeric_data1, get_numeric_data2, get_numeric_data3
import time
from imageProcessor import ImageProcessor

# Debug için dosya yolunu kontrol et
print(f"Çalışma dizini: {os.getcwd()}")

# Tema ayarları
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Root penceresini oluştur
root = ctk.CTk()
root.title("Sky Pulse Management")

# `selected_class` değişkenini root penceresinden sonra oluştur
selected_class = ctk.StringVar(value="Default")

# Video yolu (doğrudan buraya ekledim)
video_path = "/Users/ardaaydin/Desktop/Skypulse/arabalar2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video açılamadı! Programdan çıkılıyor...")
    sys.exit()

# YOLO modelini yükle
image_processor = ImageProcessor("yolo11n.pt")

detected_people = {}
frame_count = 0

# Global değişkenler ekle (dosyanın başına)
prev_frame_time = 0
new_frame_time = 0

def restart_program_with_class(selected):
    """Seçilen sınıfa göre programı yeniden başlatır."""
    current_class = selected_class.get()
    if current_class != selected:
        selected_class.set(selected)  # Seçilen sınıfı güncelle
        root.quit()  # GUI'yi sonlandır
        root.destroy()  # GUI'yi yok et
        os.execl(sys.executable, sys.executable, *sys.argv)  # Programı yeniden başlat


def buttonOperations(frame, label):
    button1 = ctk.CTkButton(frame, text="Yaklaş", command=lambda: on_button_click(1, label), fg_color="#006400", hover_color="green", text_color="white")
    button1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    button2 = ctk.CTkButton(frame, text="Farklı Açı Bak", command=lambda: on_button_click(2, label), fg_color="#FF8C00", hover_color="orange", text_color="white")
    button2.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    button3 = ctk.CTkButton(frame, text="Saldır", command=lambda: on_button_click(3, label), fg_color="#8B0000", hover_color="red", text_color="white")
    button3.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    exit_button = ctk.CTkButton(frame, text="Çıkış", command=close_program, fg_color="#1C1C1C", hover_color="gray", text_color="white")
    exit_button.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

    # Seçim Menüsünü Yaklaş butonunun yanına koy
    selection_menu = ctk.CTkOptionMenu(frame, values=["Default", "Person", "Car"], variable=selected_class, command=restart_program_with_class)
    selection_menu.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")


def on_button_click(frame_number, label):
    label.configure(text=f"Frame {frame_number} butonuna tıklandı!")


def updateVideo(cap, canvas, label):
    global frame_count
    try:
        frame_count += 1
        # Her 2 karede bir işlem yap
        if frame_count % 2 != 0:  
            return canvas.after(1, updateVideo, cap, canvas, label)

        ret, frame = cap.read()
        if not ret:
            print("Video frame'i okunamadı")
            return

        detected_people.clear()
        frame, classified_objects = image_processor.process_frame(frame, selected_class.get())
        if frame is not None:
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            canvas.create_image(0, 0, image=photo, anchor="nw")
            canvas.image = photo

            label_text = ""
            for idx, (obj, confidence, track_id) in enumerate(classified_objects.get(selected_class.get(), []), start=1):
                detected_people[f"{selected_class.get()}{idx}"] = (obj, confidence)
                label_text += f"ID:{int(track_id)} {selected_class.get().capitalize()}{idx}: Confidence {confidence:.2f}\n"

            label.configure(text=label_text)
    except Exception as e:
        print(f"Hata oluştu: {e}")

    canvas.after(1, updateVideo, cap, canvas, label)


def update_values(values_label):
    value1 = get_numeric_data1()
    value2 = get_numeric_data2()
    value3 = get_numeric_data3()
    
    values_text = f"Value1: {value1}\nValue2: {value2}\nValue3: {value3}"
    values_label.configure(text=values_text)
    
    # 500 ms (0.5 saniye) sonra tekrar çağır
    values_label.after(500, update_values, values_label)


def createGUI(root):
    root.grid_columnconfigure(0, weight=2, uniform="column")
    root.grid_columnconfigure(1, weight=1, uniform="column")
    root.grid_rowconfigure(0, weight=3, uniform="row")
    root.grid_rowconfigure(1, weight=1, uniform="row")

    frame1 = ctk.CTkFrame(root, corner_radius=10, border_width=2, border_color="red")
    frame1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    frame2 = ctk.CTkFrame(root, corner_radius=10, border_width=2, border_color="blue")
    frame2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

    # Logo ve değerleri frame2'ye ekle
    try:
        logo_img = Image.open("main.png")
        logo_img = logo_img.resize((200, 200))
        logo_photo = ctk.CTkImage(light_image=logo_img, dark_image=logo_img, size=(200, 200))
        logo_label = ctk.CTkLabel(frame2, image=logo_photo, text="")
        logo_label.image = logo_photo
        logo_label.pack(pady=20)

        # Değerler için label oluştur
        values_label = ctk.CTkLabel(frame2, text="", text_color="white")
        values_label.pack(pady=10)
        
        # Değerleri güncellemeye başla
        update_values(values_label)

        # Seçenek menüsü
        options_label = ctk.CTkLabel(frame2, text="Saldırma Yöntemi", text_color="white")
        options_label.pack(pady=10)
        options_menu = ctk.CTkOptionMenu(frame2, values=["Düz Saldır", "Zikzak çiz", "Dönerek Saldır"])
        options_menu.pack(pady=5)

    except Exception as e:
        print(f"Logo yüklenirken hata oluştu: {e}")

    frame3 = ctk.CTkFrame(root, corner_radius=10, border_width=2, border_color="green")
    frame3.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

    frame4 = ctk.CTkFrame(root, corner_radius=10, border_width=2, border_color="yellow")
    frame4.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

    canvas = ctk.CTkCanvas(frame1)
    canvas.pack(fill="both", expand=True)

    label3 = ctk.CTkLabel(frame3, text="", text_color="white")
    label3.pack(pady=20)

    buttonOperations(frame4, label3)

    updateVideo(cap, canvas, label3)



def close_program():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()


createGUI(root)
root.mainloop()