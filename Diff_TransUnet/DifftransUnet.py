"""
-*- coding: utf-8 -*-
@Project: newwork20241004
@File    : Unet015.py
@Author  : Yi-ze
@Time    : 2025-02-12 11:09:05
---- 👇 ♻注入☯灵力♻ 👇----
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from DiffTransformer.multihead_diffattn1 import DiffTransformerLayer
from DiffTransformer.multihead_diffattn2 import DiffTransformerLayer2

'''
创新的 difftrans unet
'''


class LocalAttention(nn.Module):
    ''' attention based on local importance'''

    def __init__(self, c):
        super().__init__()
        self.conv0 = nn.Conv2d(c, c, kernel_size=7, stride=3, padding=3, groups=c)
        self.transformer_layer = DiffTransformerLayer2(c)
        self.conv2 = nn.Sequential(nn.Conv2d(c*2, c, 3, padding=1), nn.ReLU())

    def forward(self, x):
        out = self.conv0(x)
        batch_size, channels, height, width = out.shape
        out = out.view(batch_size, channels, -1).permute(0, 2, 1)
        out = self.transformer_layer(out)
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)
        out= F.interpolate(out, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        out = self.conv2(torch.cat((out, x), dim=1))
        return out

###############################################################################################################


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class DoubleConv1(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            LocalAttention(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down1(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv1(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)  # //为整数除法

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)  # //为整数除法

        self.conv = DoubleConv1(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Diff_TransUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Diff_TransUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down1(256, 512)
        self.down4 = Down(512, 512)
        self.transformer_layer = DiffTransformerLayer(512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up1(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        batch_size, channels, height, width = x5.shape
        x5_flattened = x5.view(batch_size, channels, -1).permute(0, 2, 1)
        x5_transformed = self.transformer_layer(x5_flattened)
        x5_restored = x5_transformed.permute(0, 2, 1).view(batch_size, channels, height,width)
        x = self.up1(x5_restored, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    model = Diff_TransUnet(3, 5)
    a = torch.rand([8, 3, 256, 256])
    out = model(a)
    print(out.size())

