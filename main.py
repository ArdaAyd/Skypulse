import customtkinter as ctk
import cv2
import sys
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import argparse

# Tema ayarları
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Argümanları yakala
parser = argparse.ArgumentParser()
parser.add_argument("--selected_class", type=str, default="Default", help="Select class: Default, person or car")
args = parser.parse_args()

# Root penceresini oluştur
root = ctk.CTk()
root.title("Sky Pulse Management")

# Seçenekler
selected_class = ctk.StringVar(value=args.selected_class)

# Kamera ayarları
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı! Programdan çıkılıyor...")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# YOLO modelini yükle
model = YOLO("yolo11n.pt")

detected_objects = {}
frame_count = 0


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

    selection_menu = ctk.CTkOptionMenu(frame, values=["Default", "person", "car"], variable=selected_class, command=restart_program_with_class)
    selection_menu.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")


def on_button_click(frame_number, label):
    label.configure(text=f"Frame {frame_number} butonuna tıklandı!")


def updateVideo(cap, canvas, label):
    global frame_count
    try:
        frame_count += 1
        if frame_count % 5 != 0:  # Her 5 karede bir işlem yap
            return canvas.after(30, updateVideo, cap, canvas, label)

        detected_objects.clear()
        frame, classified_objects = process_frame(cap)
        if frame is not None:
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            canvas.create_image(0, 0, image=photo, anchor="nw")
            canvas.image = photo

            label_text = ""
            for idx, (obj, confidence) in enumerate(classified_objects.get(selected_class.get(), []), start=1):
                detected_objects[f"{selected_class.get()}{idx}"] = (obj, confidence)
                label_text += f"{selected_class.get().capitalize()}{idx}: Confidence {confidence:.2f}\n"

            label.configure(text=label_text)
    except Exception as e:
        print(f"Hata oluştu: {e}")

    canvas.after(30, updateVideo, cap, canvas, label)


def createGUI(root):
    root.grid_columnconfigure(0, weight=2, uniform="column")
    root.grid_columnconfigure(1, weight=1, uniform="column")
    root.grid_rowconfigure(0, weight=3, uniform="row")
    root.grid_rowconfigure(1, weight=1, uniform="row")

    frame1 = ctk.CTkFrame(root, corner_radius=10, border_width=2, border_color="red")
    frame1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    frame2 = ctk.CTkFrame(root, corner_radius=10, border_width=2, border_color="blue")
    frame2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

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


def process_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None, {}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if selected_class.get() == "Default":
        class_idx = None
    elif selected_class.get() == "person":
        class_idx = 0
    elif selected_class.get() == "car":
        class_idx = [2, 3, 5, 7]
    else:
        return None, {}

    results = model.track(rgb_frame, classes=class_idx, persist=False)

    output_frame = results[0].plot()

    classified_objects = {}
    if class_idx is not None:
        for result in results[0].boxes:
            cls = result.cls.item()
            if cls in (class_idx if isinstance(class_idx, list) else [class_idx]):
                if selected_class.get() not in classified_objects:
                    classified_objects[selected_class.get()] = []
                classified_objects[selected_class.get()].append((result.xyxy.numpy(), result.conf.item()))

    return output_frame, classified_objects


def close_program():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()


createGUI(root)
root.mainloop()
