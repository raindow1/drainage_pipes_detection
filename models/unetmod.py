import torch
import torch.nn as nn


# Модель UNet
class UNetSM(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(UNetSM, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.upconv.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._block(feature * 2, feature))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        encs = []

        # Encoder path
        for encode in self.encoder:
            x = encode(x)
            encs.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        encs = encs[::-1]
        for i in range(len(self.upconv)):
            x = self.upconv[i](x)
            enc_skip = encs[i]
            if x.shape != enc_skip.shape:
                x = torch.nn.functional.interpolate(x, size=enc_skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((enc_skip, x), dim=1)
            x = self.decoder[i](x)

        return self.final(x)

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
