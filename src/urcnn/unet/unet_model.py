""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


# UNet(n_channels=1, n_classes=1, bilinear=True)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        # super(UNet, self).__init__()  # python2.x
        super().__init__()  # python2.x
        self.n_channels = n_channels  # 输入通道 1
        self.n_classes = n_classes  # 种类数 1
        self.bilinear = bilinear  # 是否使用双线性插值 是

        """(convolution => [BN] => ReLU) * 2"""
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1  # 2
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.out_channels = 1024 // factor

    def forward(self, x, with_backbone_feature_map=True):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        backnone_feature_map = x5
        logits = self.outc(x)
        if with_backbone_feature_map:
            return logits, backnone_feature_map
        else:
            return logits
