import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Residual_3D_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual_3D_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        kernal_size = (3, 3, 3)
        stride = (1, 1, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=kernal_size,
                      stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(self.out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=kernal_size,
                      stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(self.out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):

        x = self.conv1(x)
        output = x + self.conv2(x)
        return output


class Residual_2D_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual_2D_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        kernal_size = (3, 3, 1)
        stride = (1, 1, 1)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=kernal_size,
                      stride=stride, padding=(1, 1, 0), bias=False),
            nn.InstanceNorm3d(self.out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=kernal_size,
                      stride=stride, padding=(1, 1, 0), bias=False),
            nn.InstanceNorm3d(self.out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):

        x = self.conv1(x)
        output = x + self.conv2(x)
        return output


class upsample(nn.Module):
    def __init__(self, scale, channels, out_channel):
        super(upsample, self).__init__()
        self.conv = nn.Conv3d(channels, out_channel, (3, 3, 3), padding=1)
        self.scale = nn.Upsample(scale_factor=scale)

    def forward(self, x):
        x = self.conv(x)
        return self.scale(x)


class Upsample_block(nn.Module):
    def __init__(self, channel):
        super(Upsample_block, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels=channel, out_channels=channel,
                                       kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        return self.conv(x)


class U2_5net(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(U2_5net, self).__init__()

        self.conv_0 = Residual_3D_Block(in_channel, mid_channel)
        self.conv_1 = Residual_3D_Block(mid_channel, mid_channel * 2)
        self.conv_2 = Residual_3D_Block(mid_channel * 2, mid_channel * 4)
        self.conv_3 = Residual_3D_Block(mid_channel * 4, mid_channel * 8)
        self.conv_4 = Residual_2D_Block(mid_channel * 8, mid_channel * 16)
        self.conv_5 = Residual_2D_Block(mid_channel * 16, mid_channel * 16)
        self.conv_6 = Residual_2D_Block(mid_channel * 16, mid_channel * 16)
        self.conv_7 = Residual_2D_Block(mid_channel * 16, mid_channel * 16)
        self.pool = nn.MaxPool3d((2, 2, 2))

        self.re_conv_6 = Residual_2D_Block(mid_channel * 32, mid_channel * 16)
        self.upsample_6 = Upsample_block(mid_channel * 16)
        self.re_conv_5 = Residual_2D_Block(mid_channel * 32, mid_channel * 16)
        self.upsample_5 = Upsample_block(mid_channel * 16)
        self.re_conv_4 = Residual_2D_Block(mid_channel * 32, mid_channel * 16)
        self.upsample_4 = Upsample_block(mid_channel * 16)
        self.re_conv_3 = Residual_3D_Block(mid_channel * 24, mid_channel * 8)
        self.upsample_3 = Upsample_block(mid_channel * 8)
        self.re_conv_2 = Residual_3D_Block(mid_channel * 12, mid_channel * 4)
        self.upsample_2 = Upsample_block(mid_channel * 4)
        self.re_conv_1 = Residual_3D_Block(mid_channel * 6, mid_channel * 2)
        self.out = Residual_3D_Block(mid_channel * 2, out_channel)

        self.side2 = upsample(scale=(2, 2, 2), channels=mid_channel*4, out_channel=out_channel)
        self.side3 = upsample(scale=(4, 4, 4), channels=mid_channel*8, out_channel=out_channel)
        self.side4 = upsample(scale=(8, 8, 8), channels=mid_channel*16, out_channel=out_channel)
        self.side5 = upsample(scale=(16, 16, 16), channels=mid_channel*16, out_channel=out_channel)

    def forward(self, x):
        hx = x
        hx_in = self.conv_0(hx)
        hx_1 = self.conv_1(hx_in)
        hx = self.pool(hx_1)

        hx_2 = self.conv_2(hx)
        hx = self.pool(hx_2)

        hx_3 = self.conv_3(hx)
        hx = self.pool(hx_3)

        hx_4 = self.conv_4(hx)
        hx = self.pool(hx_4)

        hx_5 = self.conv_5(hx)
        hx = self.pool(hx_5)

        hx_6 = self.conv_6(hx)
        hx_7 = self.conv_7(hx_6)

        hx_6d = self.re_conv_6(torch.cat((hx_7, hx_6), 1))
        hx_6d_up = self.upsample_6(hx_6d)

        hx_5d = self.re_conv_5(torch.cat((hx_6d_up, hx_5), 1))
        d_5 = self.side5(hx_5d)
        hx_5d_up = self.upsample_5(hx_5d)

        hx_4d = self.re_conv_4(torch.cat((hx_5d_up, hx_4), 1))
        d_4 = self.side4(hx_4d)
        hx_4d_up = self.upsample_4(hx_4d)

        hx_3d = self.re_conv_3(torch.cat((hx_4d_up, hx_3), 1))
        d_3 = self.side3(hx_3d)
        hx_3d_up = self.upsample_3(hx_3d)

        hx_2d = self.re_conv_2(torch.cat((hx_3d_up, hx_2), 1))
        d_2 = self.side2(hx_2d)
        hx_2d_up = self.upsample_2(hx_2d)

        hx_1d = self.re_conv_1(torch.cat((hx_2d_up, hx_1), 1))

        return F.sigmoid(self.out(hx_1d)), [F.sigmoid(d_2), F.sigmoid(d_3), F.sigmoid(d_4), F.sigmoid(d_5)]