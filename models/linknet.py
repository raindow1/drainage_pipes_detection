import torch
import torch.nn as nn
from torchvision import models


class DecoderBlock(nn.Module):
    """Блок декодера LinkNet:
    Состоит из 1x1 свертки -> ConvTranspose2d (upsampling) -> 1x1 свертки"""

    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            # 1x1 свертка для уменьшения размерности
            nn.Conv2d(in_channels, middle_channels, kernel_size=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),

            # Transposed convolution для увеличения разрешения в 2 раза
            nn.ConvTranspose2d(
                middle_channels, middle_channels,
                kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),

            # Финальная 1x1 свертка
            nn.Conv2d(middle_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LinkNet(nn.Module):
    """Архитектура LinkNet с ResNet101 в качестве энкодера"""

    def __init__(self, num_classes=1, in_channels=3):
        super(LinkNet, self).__init__()

        # Инициализация предобученного ResNet101
        resnet = models.resnet101(pretrained=True)

        # Входной блок (адаптирован из ResNet)
        self.in_block = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # Энкодер (заимствуем слои из ResNet)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Декодер
        self.decoder4 = DecoderBlock(2048, 512, 1024)
        self.decoder3 = DecoderBlock(1024, 256, 512)
        self.decoder2 = DecoderBlock(512, 128, 256)
        self.decoder1 = DecoderBlock(256, 64, 64)

        # Финальный upsampling + классификатор
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1/2 -> исходный размер
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Проход через энкодер
        x = self.in_block(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Проход через декодер с skip-connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Финальный upsampling до исходного размера
        out = self.final_deconv(d1) 
        return out