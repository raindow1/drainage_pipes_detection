# import numpy as np
#
# # Загрузка
# masks_part1 = np.load('mask_ctg1.npy')
# masks_part2 = np.load('mask_ctg2.npy')
#
# # Проверка размеров
# assert masks_part1.shape[1:] == masks_part2.shape[1:], "Размеры не совпадают!"
#
# # Объединение
# combined_masks = np.concatenate([masks_part1, masks_part2], axis=0)
#
# # Сохранение
# np.save('combined_masks.npy', combined_masks)
# print(f"Объединено {len(combined_masks)} масок. Форма: {combined_masks.shape}")

import numpy as np
import cv2
import os

# Загрузка масок
masks = np.load('../combined_masks.npy')
output_dir = '../masks'
os.makedirs(output_dir, exist_ok=True)

# Конвертация
for i in range(masks.shape[0]):
    mask = masks[i]

    # Если маски one-hot (N, H, W, C)
    if mask.ndim == 4:
        mask = np.argmax(mask, axis=-1)

    # Конвертация в uint8
    mask = mask.astype(np.uint8)

    # Для бинарных масок: 0 -> 0, 1 -> 255
    if np.max(mask) == 1:
        mask = mask * 255

    # Сохранение
    cv2.imwrite(f'{output_dir}/mask_{i}.png', mask)

print(f"Сохранено {masks.shape[0]} масок в {output_dir}")