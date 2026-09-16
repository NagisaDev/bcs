from ultralytics import YOLO

# ВНИМАНИЕ: Берем модель с суффиксом -seg (для сегментации)
model = YOLO('yolov8n-seg.pt')

if __name__ == '__main__':
    # Запуск обучения
    results = model.train(
        data='dataset_seg/data.yaml',  # Путь к нашему новому датасету с масками
        epochs=50,  # 50 эпох для хорошего контура
        imgsz=640,
        batch=8,  # Сегментация жрет чуть больше памяти. Если вылетит ошибка CUDA Out of Memory, ставь 4
        workers=2,
        plots=True,

        # Аугментация (чтобы не пугалась схем)
        hsv_h=0.1,
        hsv_s=0.5,
        hsv_v=0.5,

        device='0'  # Используем твою GTX 1080 иначе  ПОМЕНЯЯЯЯЙ  на  'cpu'
    )

    print("\n[INFO] Обучение сегментации завершено!")
    print("[INFO] Ищи веса в папке runs/segment/train/weights/best.pt")