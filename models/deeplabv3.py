import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DeepLabV3PlusResNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabV3PlusResNet50, self).__init__()

        # Инициализация предобученного ResNet50 в качестве энкодера
        resnet50 = models.resnet50(pretrained=True)

        # Переопределение слоев ResNet для удобства доступа
        self.layer0 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool
        )

        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        # Модуль ASPP для обработки features высокого уровня
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # Декодер для объединения features высокого и низкого уровней
        self.decoder = Decoder(low_level_channels=256, num_classes=num_classes)

        # Инициализация весов новых слоев
        self._init_weight()

    def forward(self, x):
        # Энкодер (извлечение features на разных уровнях)
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Обработка features высокого уровня через ASPP
        aspp_out = self.aspp(x4)

        # Декодер: объединяет ASPP output и low-level features
        decoder_out = self.decoder(aspp_out, x1)

        # Интерполяция к исходному размеру изображения
        output = F.interpolate(decoder_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

    def _init_weight(self):
        # Инициализация весов для новых слоев
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling модуль:
    Обрабатывает features на разных масштабах с помощью сверток с разными dilation rates"""

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        # 1x1 свертка
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 3x3 свертки с разными dilation rates
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Global Average Pooling ветка
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Сводит к 1x1
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Финалная свертка для объединения всех веток
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Регуляризация
        )

    def forward(self, x):
        h, w = x.shape[2:]  # Сохраняем размер features

        # Применяем все параллельные ветки
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # GAP ветка (1x1 -> интерполяция к исходному размеру)
        img_pool = self.image_pooling(x)
        img_pool = F.interpolate(img_pool, size=(h, w), mode='bilinear', align_corners=False)

        # Конкатенация всех веток
        out = torch.cat([x1, x2, x3, x4, img_pool], dim=1)
        out = self.conv_out(out)

        return out


class Decoder(nn.Module):
    """Декодер DeepLabv3+:
    Объединяет high-level features (из ASPP) и low-level features (из ранних слоев)"""

    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()

        # Свертка для low-level features (уменьшение каналов)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Блоки обработки объединенных features
        self.conv1 = nn.Sequential(
            nn.Conv2d(304, 256, 3, stride=1, padding=1, bias=False),  # 256+48=304
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Финальная свертка для получения маски
        self.final_conv = nn.Conv2d(256, num_classes, 1)  # 1x1 свертка

    def forward(self, x, low_level_feat):
        # Обработка low-level features
        low_level_feat = self.low_level_conv(low_level_feat)

        # Интерполяция high-level features к размеру low-level
        h, w = low_level_feat.shape[2:]
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        # Объединение features
        x = torch.cat([x, low_level_feat], dim=1)

        # Обработка объединенных features
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_conv(x)

        return x