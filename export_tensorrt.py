from ultralytics import YOLO
import torch

def export_model_to_tensorrt(model_path):
    # YOLO modelini yükle
    model = YOLO(model_path)
    
    # TensorRT için export ayarları
    model.export(format='engine',  # TensorRT engine formatı
                device='cuda',     # CUDA cihazı kullan
                half=True,         # FP16 precision kullan
                simplify=True,     # ONNX modelini basitleştir
                workspace=4,       # GB cinsinden workspace
                verbose=True)      # Export detaylarını göster

if __name__ == "__main__":
    model_path = "best.pt"  # Model yolunu ayarlayın
    export_model_to_tensorrt(model_path) 