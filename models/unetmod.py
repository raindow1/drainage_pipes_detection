import torch
import torch.nn as nn


class UNetSM(nn.Module):
    """U-Net архитектура для семантической сегментации с динамической настройкой глубины"""

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(UNetSM, self).__init__()

        # Инициализация модулей
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()

        # Построение энкодера
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature

        # Центральный блок (bottleneck) - самый глубокий уровень
        self.bottleneck = self._block(features[-1], features[-1] * 2)

        # Построение декодера (в обратном порядке)
        for feature in reversed(features):
            # Транспонированная свертка для увеличения разрешения
            self.upconv.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            # Блок обработки после конкатенации с skip-connection
            self.decoder.append(self._block(feature * 2, feature))

        # Финальная 1x1 свертка для получения выходной маски
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Сохраняем выходы энкодера для skip-connections
        encs = []

        # Проход через энкодер
        for encode in self.encoder:
            x = encode(x)
            encs.append(x)
            x = self.pool(x)

        # Центральный блок (самое низкое разрешение)
        x = self.bottleneck(x)

        # Разворачиваем список для соответствия уровням декодера
        encs = encs[::-1]

        # Проход через декодер
        for i in range(len(self.upconv)):
            x = self.upconv[i](x)

            # Получаем соответствующий skip-connection
            enc_skip = encs[i]

            # Проверка размеров (на случай нечетных размеров изображения)
            if x.shape != enc_skip.shape:
                x = torch.nn.functional.interpolate(
                    x, size=enc_skip.shape[2:],
                    mode='bilinear', align_corners=True
                )

            # Конкатенация с skip-connection
            x = torch.cat((enc_skip, x), dim=1)

            # Применяем блок сверток
            x = self.decoder[i](x)

        # Финальная свертка
        return self.final(x)

    @staticmethod
    def _block(in_channels, out_channels):
        """Базовый блок из двух сверточных слоев с BatchNorm и ReLU
        """
        return nn.Sequential(
            # Первая свертка 3x3
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Вторая свертка 3x3
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )