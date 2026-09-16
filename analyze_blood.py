import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os

model_path = r'runs\segment\train\weights\best.pt'
model = YOLO(model_path)
classes_names = model.names

img_name = "С1.png"
img_path = f"im2/{img_name}"

if not os.path.exists(img_path):
    print(f"\n[ОШИБКА] Файл '{img_path}' не найден!")
    exit()

img = cv2.imread(img_path)
#img = normalize_image(img)

# 3. ПОИСК КЛЕТОК (Инференс)
results = model(img, conf=0.30, iou=0.25, imgsz=1024)

cells_data = []

# 4. ОБРАБОТКА МАСКОК (Контуров)
if results[0].masks is not None:
    boxes = results[0].boxes
    masks = results[0].masks

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = classes_names[cls_id]

        polygon = masks.xy[i]
        area = cv2.contourArea(polygon) if len(polygon) > 2 else 0
        x, y, w, h = cv2.boundingRect(np.array(polygon).astype(np.float32))
        aspect_ratio = w / h if h > 0 else 0

        cells_data.append({
            'Class': class_name,
            'Confidence': round(conf, 2),
            'Real_Area_px2': round(area, 1),
            'Aspect_Ratio': round(aspect_ratio, 2)
        })

# ОТРИСОВКА YOLO сама
annotated_img = results[0].plot()

# ЭКСПОРТ ТАБЛИЦЫ
df = pd.DataFrame(cells_data)

if not df.empty:
    print("\n--- СТАТИСТИКА НАХОДОК ---")
    print(df['Class'].value_counts())
    df.to_csv("blood_segmentation_results.csv", index=False)
    print("\n[INFO] Таблица морфологии сохранена в 'blood_segmentation_results.csv'")
else:
    print("\n[ВНИМАНИЕ] Клеток не найдено.")

h, w, _ = annotated_img.shape
if h > 900:
    annotated_img = cv2.resize(annotated_img, (int(w * 900 / h), 900))

cv2.imshow("Instance Segmentation - Morphological Analysis", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()