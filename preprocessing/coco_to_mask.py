from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Путь к JSON-файлу с аннотациями COCO
annotation_file = "../instances_Train.json"

# Путь к директории с изображениями
image_dir = "../google_earth_images"

# Путь для сохранения масок
output_mask_dir = "../google_masks"
os.makedirs(output_mask_dir, exist_ok=True)

# Загружаем аннотации COCO
coco = COCO(annotation_file)

# Получаем список всех изображений
image_ids = coco.getImgIds()

# Проходим по каждому изображению
for img_id in image_ids:
    # Загружаем информацию об изображении
    img_info = coco.loadImgs(img_id)[0]
    image_path = os.path.join(image_dir, img_info['file_name'])

    # Загружаем изображение (для визуализации, если нужно)
    image = cv2.imread(image_path)
    height, width = img_info['height'], img_info['width']

    # Создаем пустую маску для изображения
    mask = np.zeros((height, width), dtype=np.uint8)

    # Получаем все аннотации для этого изображения
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # Проходим по всем аннотациям
    for ann in anns:
        # Получаем сегментацию
        if 'segmentation' in ann:
            rle = coco.annToMask(ann)
            mask[rle == 1] = 255  # Задаем белый цвет для маски (255)

    # Сохраняем маску
    mask_filename = os.path.join(output_mask_dir, f"{img_id}.png")
    cv2.imwrite(mask_filename, mask)
    print(f"Маска сохранена: {mask_filename}")

