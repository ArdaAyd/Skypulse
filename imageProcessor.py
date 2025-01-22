from ultralytics import YOLO
import cv2

# YOLO modelini yükle
model = YOLO("yolo5n.pt")

# Varsayılan sınıf seçimi (Default: hiçbir sınıf)
selected_class = "Default"

def process_frame(cap, frame_count):
    ret, frame = cap.read()
    if not ret:
        return None, {}

    # İşlem yükünü azaltmak için her 10 karede bir işlem yap
    if frame_count % 10 != 0:
        return frame, {}

    # YOLO ile çalışmak için BGR'den RGB'ye dönüştür
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Sınıf seçimine göre işlem yap
    if selected_class == "Default":
        class_idx = None
    elif selected_class == "person":
        class_idx = 0
    elif selected_class == "car":
        class_idx = [2, 3, 5, 7]
    else:
        return frame, {}

    # YOLO modelinde inference
    results = model.track(rgb_frame, classes=class_idx, persist=False)

    # Sonuçları kareye çizin
    output_frame = results[0].plot()

    # Sınıflandırılan nesneleri ayıkla
    classified_objects = {}
    if class_idx is not None:
        for result in results[0].boxes:
            cls = result.cls.item()
            if cls in (class_idx if isinstance(class_idx, list) else [class_idx]):
                class_name = "person" if cls == 0 else "car"
                if class_name not in classified_objects:
                    classified_objects[class_name] = []
                classified_objects[class_name].append((result.xyxy.numpy(), result.conf.item()))

    return output_frame, classified_objects

# Örnek kullanım
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
selected_class = input("Lütfen sınıf seçin (Default, person, car): ").strip()

while True:
    frame_count += 1

    # Kare işle
    frame, classified_objects = process_frame(cap, frame_count)

    if frame is not None:
        cv2.imshow("YOLO Detection", frame)

    # Algılanan nesneleri yazdır
    for obj_class, obj_list in classified_objects.items():
        for obj, confidence in obj_list:
            print(f"Class: {obj_class}, Confidence: {confidence:.2f}")

    # 'q' tuşu ile çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
