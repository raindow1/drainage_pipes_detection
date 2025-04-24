import numpy as np
from PIL import Image
import os

# Функция для сохранения массива numpy как изображения
def save_npy_as_image(npy_array, output_path):
    # Преобразуем массив в изображение
    image = Image.fromarray(npy_array.astype('uint8'))
    # Сохраняем изображение в формате JPEG
    image.save(output_path)

# Загрузка данных из .npy файлов
images_npy = np.load('../data_small.npy')  # Загружаем массив с изображениями
# masks_npy = np.load('../combined_masks.npy')   # Загружаем массив с масками

# Создаем директории для сохранения изображений и масок, если они не существуют
os.makedirs('../images', exist_ok=True)
# os.makedirs('../masks', exist_ok=True)

# Сохранение изображений
for i, image in enumerate(images_npy):
    save_npy_as_image(image, f'../images/image_{i}.jpg')

# # Сохранение масок
# for i, mask in enumerate(masks_npy):
#     # Если маска одноканальная, преобразуем ее в трехканальную (для сохранения как JPEG)
#     if mask.ndim == 2:
#         mask = np.stack([mask] * 3, axis=-1)
#     save_npy_as_image(mask, f'../masks/mask_{i}.jpg')

print("Изображения и маски успешно сохранены.")