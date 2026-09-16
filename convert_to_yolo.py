import os
import cv2

images_dir = "images"
labels_dir = "labels"

class_map = {'wbc': 0, 'rbc': 1, 'platelets': 2}

# Счетчики для проверки
converted_count = 0
skipped_count = 0

for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    label_path = os.path.join(labels_dir, label_file)

    # МЕНЯЕМ .txt НА .png
    img_name = label_file.replace(".txt", ".png")
    img_path = os.path.join(images_dir, img_name)

    # Проверяем, существует ли картинка
    if not os.path.exists(img_path):
        print(f"[-] Пропущен {label_file}: не найдена картинка {img_path}")
        skipped_count += 1
        continue

    # Читаем размеры картинки
    img = cv2.imread(img_path)
    if img is None:
        print(f"[-] Ошибка чтения картинки {img_path}")
        continue

    img_h, img_w, _ = img.shape

    with open(label_path, "r") as file:
        lines = file.readlines()

    yolo_lines = []
    already_converted = False

    for line in lines:
        data = line.strip().split()

        # Если файл уже в формате YOLO (5 чисел), то пропускаем
        if len(data) == 5:
            already_converted = True
            break

        if len(data) < 8:
            continue

        class_str = data[0].lower()
        if class_str not in class_map:
            continue

        class_id = class_map[class_str]

        x1 = float(data[4])
        y1 = float(data[5])
        x2 = float(data[6])
        y2 = float(data[7])

        box_w = x2 - x1
        box_h = y2 - y1
        x_center = x1 + (box_w / 2)
        y_center = y1 + (box_h / 2)

        x_center /= img_w
        y_center /= img_h
        box_w /= img_w
        box_h /= img_h

        x_center, y_center = min(max(x_center, 0), 1), min(max(y_center, 0), 1)
        box_w, box_h = min(max(box_w, 0), 1), min(max(box_h, 0), 1)

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

    if already_converted:
        print(f"[~] Файл {label_file} уже был в формате YOLO")
        continue

    # Перезаписываем файл
    if yolo_lines:
        with open(label_path, "w") as file:
            file.writelines(yolo_lines)
        print(f"[+] Конвертирован {label_file}")
        converted_count += 1

print("\n--- ИТОГИ ---")
print(f"Успешно конвертировано: {converted_count}")
print(f"Пропущено (нет картинки): {skipped_count}")