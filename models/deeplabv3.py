import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DeepLabV3PlusResNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabV3PlusResNet50, self).__init__()

        resnet50 = models.resnet50(pretrained=True)
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

        self.aspp = ASPP(in_channels=2048, out_channels=256)

        self.decoder = Decoder(low_level_channels=256, num_classes=num_classes)

        self._init_weight()

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)  # 1/4
        x1 = self.layer1(x0)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        # ASPP
        aspp_out = self.aspp(x4)

        # Decoder with low-level features from layer1
        decoder_out = self.decoder(aspp_out, x1)


        output = F.interpolate(decoder_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

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

        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        h, w = x.shape[2:]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # Image pooling branch
        img_pool = self.image_pooling(x)
        img_pool = F.interpolate(img_pool, size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate all branches
        out = torch.cat([x1, x2, x3, x4, img_pool], dim=1)
        out = self.conv_out(out)

        return out


class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(304, 256, 3, stride=1, padding=1, bias=False),
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

        self.final_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_conv(low_level_feat)

        # Upsample ASPP output
        h, w = low_level_feat.shape[2:]
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate with low-level features
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_conv(x)

        return x