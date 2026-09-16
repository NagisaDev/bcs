import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# 1. НАСТРОЙКИ СТРАНИЦЫ
st.set_page_config(page_title="Blood Cell Segmentation", layout="wide")
st.title("🔬 AI Анализ морфологии крови")
st.markdown("Система автоматической сегментации клеток: эритроциты, лейкоциты, тромбоциты.")


# 2. ЗАГРУЗКА МОДЕЛИ (Кэшируем, чтобы не загружать видеокарту при каждом клике)
@st.cache_resource
def load_model():
    # ВНИМАНИЕ: Проверь, что путь к твоей лучшей модели указан верно!
    return YOLO(r'runs\segment\train\weights\best.pt')


model = load_model()
classes_names = model.names

# 3. БОКОВОЕ МЕНЮ (Настройки)
st.sidebar.header("Настройки нейросети")
conf_threshold = st.sidebar.slider("Порог уверенности (Confidence)", min_value=0.05, max_value=0.95, value=0.30,
                                   step=0.05)
iou_threshold = st.sidebar.slider("Порог пересечения (IoU)", min_value=0.05, max_value=0.95, value=0.45, step=0.05)

# 4. ЗАГРУЗКА ИЗОБРАЖЕНИЯ ПОЛЬЗОВАТЕЛЕМ
uploaded_file = st.file_uploader("Загрузите снимок с микроскопа (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Конвертируем загруженный файл в формат OpenCV
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Исходное изображение", use_container_width=True)

    # 5. ЗАПУСК АНАЛИЗА
    if st.button("🚀 Запустить анализ", type="primary"):
        with st.spinner('Нейросеть обрабатывает снимок...'):

            # Инференс
            results = model(img_bgr, conf=conf_threshold, iou=iou_threshold, imgsz=1024)

            cells_data = []

            # Обработка масок
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
                        'Класс': class_name,
                        'Уверенность': round(conf, 2),
                        'Площадь (px2)': round(area, 1),
                        'Форма (W/H)': round(aspect_ratio, 2)
                    })

            # Отрисовка
            annotated_bgr = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            st.divider()

            # 6. ВЫВОД РЕЗУЛЬТАТОВ НА ЭКРАН
            col1, col2 = st.columns([3, 2])  # Делим экран на две колонки

            with col1:
                st.subheader("Результат сегментации")
                st.image(annotated_rgb, use_container_width=True)

            with col2:
                st.subheader("📊 Аналитика")
                if cells_data:
                    df = pd.DataFrame(cells_data)

                    # Подсчет количества каждого типа клеток
                    counts = df['Класс'].value_counts().reset_index()
                    counts.columns = ['Тип клетки', 'Найдено штук']
                    st.dataframe(counts, hide_index=True, use_container_width=True)

                    # Полная таблица морфологии
                    st.dataframe(df, hide_index=True, use_container_width=True)

                    # Кнопка для скачивания отчета в Excel/CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Скачать полный отчет (CSV)",
                        data=csv,
                        file_name="blood_analysis_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Клеток не найдено. Попробуйте снизить 'Порог уверенности' в боковом меню.")