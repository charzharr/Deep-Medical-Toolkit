""" bigunet2d.py
Larger U-Net PyTorch implementation from:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel


def get_model(in_channels=1, out_channels=1, bilinear_up=True, base_size=80):
    """
    Args:
        in_channels: # of input image channels.
        out_channels: # of output classes (specifically output dims).
        bilinear_up: Flag to use bilinear interpolation or upconvolution.
        base_size: Number of channels for base convolution (64 on std unet).
    """
    model = BigUNet2D(in_channels, out_channels, 
                      bilinear=bilinear_up, base_size=base_size)
    return model



### ======================================================================== ###
### * ### * ### * ### *          U-Net Components        * ### * ### * ### * ###
### ======================================================================== ###


class DoubleConv(nn.Module):
    """ (Same-Conv => [BN] => ReLU) * 2 """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# class BottleNeck(nn.Module):
#     def __init__(self, in_channels, out_channels)


class Down(nn.Module):
    """ 2x Downscale MP -> Double Conv """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """ 2x Upscale -> Double Conv """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, 
                mode='bilinear', 
                align_corners=True
            )
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels , 
                in_channels // 2, 
                kernel_size=2, 
                stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """ 1x1 Conv for Last Layer """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


### ======================================================================== ###
### * ### * ### * ### *               U-Net              * ### * ### * ### * ###
### ======================================================================== ###


class BigUNet2D(BaseModel):
    def __init__(self, n_channels, n_classes, bilinear=True, base_size=80):
        """
        Parameters
            n_channels - number of input channels in image
            n_classes - number of output channels that model will output
            bilinear (bool) - use bilinear upsampling or transposed conv
            base_size (int) - std U-Net: 64, 80 is 25% inc
        """
        super(BigUNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_size = base_size
        if self.base_size % 2 == 1:
            self.base_size -= 1  # make channels even
        print(f"  Prepping BigUNet w/base_size {self.base_size}.")

        self.inc = DoubleConv(n_channels, self.base_size)
        self.down1 = Down(self.base_size, self.base_size * 2)
        self.down2 = Down(self.base_size * 2, self.base_size * 4)
        self.down3 = Down(self.base_size * 4, self.base_size * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.base_size * 8, self.base_size * 16 // factor)
        
        self.up1 = Up(self.base_size * 16, self.base_size * 8 // factor, bilinear)
        self.up2 = Up(self.base_size * 8, self.base_size * 4 // factor, bilinear)
        self.up3 = Up(self.base_size * 4, self.base_size * 2 // factor, bilinear)
        self.up4 = Up(self.base_size * 2, self.base_size, bilinear)
        self.outc = OutConv(self.base_size, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

