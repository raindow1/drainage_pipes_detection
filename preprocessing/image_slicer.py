import os
import cv2
import numpy as np
from tqdm import tqdm  # для прогресс-бара

# Пути к данным
src_images_dir = "../google_earth_images"
src_masks_dir = "../google_masks"
dst_images_dir = "../dataset/images"
dst_masks_dir = "../dataset/masks"

# Параметры разбиения
patch_size = 256
stride = 128  # Перекрытие = patch_size - stride = 128 (50%)

# Создаем папки для сохранения
os.makedirs(dst_images_dir, exist_ok=True)
os.makedirs(dst_masks_dir, exist_ok=True)


def pad_to_size(img, target_size=256):
    """Дополняет изображение до target_size x target_size (reflect padding)."""
    h, w = img.shape[:2]
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    if len(img.shape) == 3:  # Цветное изображение
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:  # Маска (grayscale)
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')


def split_and_save(image_path, mask_path, img_name):
    # Загрузка изображения и маски
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Проверка размеров
    assert image.shape[:2] == mask.shape[:2], f"Размеры {img_name} не совпадают!"
    h, w = image.shape[:2]

    # Генерация патчей с перекрытием
    patch_id = 0
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            # Вырезаем патчи
            patch_img = image[i:i + patch_size, j:j + patch_size]
            patch_mask = mask[i:i + patch_size, j:j + patch_size]

            # Дополнение, если патч меньше 256x256
            if patch_img.shape[0] < patch_size or patch_img.shape[1] < patch_size:
                patch_img = pad_to_size(patch_img, patch_size)
                patch_mask = pad_to_size(patch_mask, patch_size)


            # Сохраняем патчи
            patch_name = f"{os.path.splitext(img_name)[0]}_{i}_{j}.png"
            cv2.imwrite(os.path.join(dst_images_dir, patch_name), patch_img)
            cv2.imwrite(os.path.join(dst_masks_dir, patch_name), patch_mask)
            patch_id += 1


# Обработка всех изображений
for img_name in tqdm(os.listdir(src_images_dir)):
    if not img_name.endswith(".png"):
        continue

    img_path = os.path.join(src_images_dir, img_name)
    mask_path = os.path.join(src_masks_dir, img_name)

    if not os.path.exists(mask_path):
        print(f"Маска для {img_name} не найдена! Пропускаем.")
        continue

    split_and_save(img_path, mask_path, img_name)

print(f"Готово! Патчи сохранены в:\n{dst_images_dir}\n{dst_masks_dir}")