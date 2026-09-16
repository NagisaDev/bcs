import cv2

img_name = "image-1.png"
img_path = f"images/{img_name}"
label_path = f"labels/{img_name.replace('.png', '.txt')}"

img = cv2.imread(img_path)
h, w, _ = img.shape

# 0 - WBC (Лейкоциты), 1 - RBC (Эритроциты), 2 - Platelets (Тромбоциты)
classes = ["WBC", "RBC", "Platelets"]
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Цвета рамок (BGR)

with open(label_path, "r") as file:
    lines = file.readlines()

# Теперь файл в формате YOLO: class_id x_center y_center width height (все от 0 до 1)
for line in lines:
    data = line.strip().split()

    # 1. Получаем класс (теперь это сразу готовая цифра, словарь class_map больше не нужен)
    class_id = int(data[0])

    # 2. Берем нормализованные координаты (от 0 до 1)
    x_center_norm = float(data[1])
    y_center_norm = float(data[2])
    width_norm = float(data[3])
    height_norm = float(data[4])

    # 3. Переводим проценты в реальные пиксели (умножаем на ширину и высоту картинки)
    x_center = x_center_norm * w
    y_center = y_center_norm * h
    box_w = width_norm * w
    box_h = height_norm * h

    # 4. Вычисляем координаты углов для OpenCV (x1, y1 - левый верхний, x2, y2 - правый нижний)
    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    # 5. Рисуем прямоугольник и подписываем
    color = colors[class_id]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, classes[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Dataset Check YOLO", img)
cv2.waitKey(0)
cv2.destroyAllWindows()