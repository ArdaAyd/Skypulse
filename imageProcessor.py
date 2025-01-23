from ultralytics import YOLO
import cv2
import argparse

# Argümanlar için parser
parser = argparse.ArgumentParser()
parser.add_argument("--selected_class", type=str, default="Default", help="Select class: Default, person or car")
parser.add_argument("--video_path", type=str, default="", help="Path to video file")
args = parser.parse_args()

# YOLO modelini yükle
model = YOLO("yolo11n-obb.pt")

# Seçilen sınıfı belirle (0: person, 2/3/5/7: car gibi)
selected_class = None
if args.selected_class == "person":
    selected_class = 0
elif args.selected_class == "car":
    selected_class = [2, 3, 5, 7]
elif args.selected_class == "Default":
    selected_class = None

# Video kaynağı belirle (video dosyası ya da kamera)
if args.video_path:
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Video dosyası açılamadı: {args.video_path}")
        exit()
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0

def process_frame(cap, frame_count):
    ret, frame = cap.read()
    if not ret:
        return None, {}

    # İş yükünü azaltmak için her 10. karede bir işlem yapıyoruz
    

    # BGR'den RGB'ye dönüştür
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO ile sınıfı belirle ve sonuçları al
    results = model.track(rgb_frame, classes=selected_class, persist=False)

    # Çıktıları çerçeveye çiz
    output_frame = results[0].plot()

    # Algılanan nesneleri sınıflandır
    classified_objects = {}
    if selected_class is not None:
        for result in results[0].boxes:
            cls = result.cls.item()
            if cls in (selected_class if isinstance(selected_class, list) else [selected_class]):
                class_name = "person" if cls == 0 else "car"
                if class_name not in classified_objects:
                    classified_objects[class_name] = []
                classified_objects[class_name].append((result.xyxy.numpy(), result.conf.item()))

    return output_frame, classified_objects

while True:
    frame_count += 1

    # Kareyi işle
    frame, classified_objects = process_frame(cap, frame_count)

    if frame is not None:
        cv2.imshow("YOLO Detection", frame)

    # Algılanan nesneleri ekrana yazdır
    for obj_class, obj_list in classified_objects.items():
        for obj, confidence in obj_list:
            print(f"Class: {obj_class}, Confidence: {confidence:.2f}")

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
