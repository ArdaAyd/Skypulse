import customtkinter as ctk
import cv2
import sys
from io import StringIO
from PIL import Image, ImageTk
from imageProcessor import process_frame
from data import get_numeric_data1, get_numeric_data2, get_numeric_data3


#theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


root = ctk.CTk()
root.title("Sky Pulse Management")
root.geometry("800x600")
cap=cv2.VideoCapture(0)

def buttonOperations(frame,label):
    button1 = ctk.CTkButton(frame, text="Yaklaş", command=lambda: on_button_click(1,label), fg_color="#006400", hover_color="green", text_color="white")
    button1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Aralara boşluk ekledik

    button2 = ctk.CTkButton(frame, text="Farklı Açı Bak", command=lambda: on_button_click(2,label), fg_color="#FF8C00", hover_color="orange", text_color="white")
    button2.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    button3 = ctk.CTkButton(frame, text="Saldır", command=lambda: on_button_click(3,label), fg_color="#8B0000", hover_color="red", text_color="white")
    button3.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
def on_button_click(frame_number,label):
    old_stdout = sys.stdout
    sys.stdout = StringIO()  # Çıktıyı StringIO'ya yönlendir

    print(f"Frame {frame_number} butonuna tıklandı!")  # Bu print çıktısını yakalayacağız

    # StringIO'dan veriyi alıp label3'e yazdır
    label.configure(text=sys.stdout.getvalue())

    # stdout'u eski haline getir
    sys.stdout = old_stdout
def updateVideo(cap, canvas):
    frame=process_frame(cap)
    if frame is not None:
        image=Image.fromarray(frame)
        photo=ImageTk.PhotoImage(image=image)
        canvas.create_image(0,0, image=photo, anchor="nw")
        canvas.image=photo

    canvas.after(10, updateVideo, cap, canvas)
def updateData(label,root):
    value1=get_numeric_data1()
    value2=get_numeric_data2()
    value3=get_numeric_data3()
    label.configure(text=f"Value1: {value1}\nValue2: {value2}\nValue3: {value3}")

    root.after(1000,updateData,label,root)
def createGUI(root):

    root.grid_columnconfigure(0, weight=2, uniform="column")  # Sol sütun genişliği daha büyük
    root.grid_columnconfigure(1, weight=1, uniform="column")  # Sağ sütun daha dar

    # Satır yapılandırması
    root.grid_rowconfigure(0, weight=3, uniform="row")  # Üst satır daha geniş
    root.grid_rowconfigure(1, weight=1, uniform="row")  # Alt satır daha dar

    # Çerçeveler
    frame1 = ctk.CTkFrame(root, corner_radius=10, border_width=2, border_color="red")
    frame1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")  # Sol üst

    frame2 = ctk.CTkFrame(root, corner_radius=10, border_width=2, border_color="blue")
    frame2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")  # Sağ üst

    frame3 = ctk.CTkFrame(root, corner_radius=10, border_width=2, border_color="green")
    frame3.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")  # Sol alt

    frame4 = ctk.CTkFrame(root, corner_radius=10, border_width=2, border_color="yellow")
    frame4.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")  # Sağ alt

    # İçerik ekleme (isteğe bağlı)
    canvas = ctk.CTkCanvas(frame1, width=400, height=400)
    canvas.pack(fill="both", expand=True)

    label2 = ctk.CTkLabel(frame2, text_color="white")
    label2.pack(pady=20)

    label3 = ctk.CTkLabel(frame3, text="", text_color="white")
    label3.pack(pady=20)

    updateVideo(cap,canvas)
    updateData(label2,root)
    buttonOperations(frame4,label3)


createGUI(root)
root.mainloop()
cap.release()
