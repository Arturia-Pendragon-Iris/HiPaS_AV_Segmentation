import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class CBRConv_3d(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, dirate=1):
        super(CBRConv_3d, self).__init__()

        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3),
                              stride=(1, 1, 1), padding=1*dirate, dilation=(dirate, dirate, dirate))
        self.bn = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CBRConv_2d(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, dirate=1):
        super(CBRConv_2d, self).__init__()

        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 1),
                              stride=(1, 1, 1), padding=(1, 1, 0), dilation=(1, 1, dirate))
        self.bn = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def upsample(src, scale=(2, 2, 2)):
    src = nn.Upsample(src, scale_factor=scale, mode='trilinear')
    return src


class RSU7(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(RSU7, self).__init__()
        self.conv_0 = CBRConv_3d(in_channel, out_channel, dirate=1)
        self.conv_1 = CBRConv_3d(out_channel, mid_channel, dirate=1)
        self.conv_2 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_3 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_4 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_5 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_6 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_7 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        self.re_conv_6 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_5 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_4 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_3 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_2 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_1 = CBRConv_3d(mid_channel * 2, out_channel, dirate=1)

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
        hx_6d_up = self.upsample(hx_6d)

        hx_5d = self.re_conv_5(torch.cat((hx_6d_up, hx_5), 1))
        hx_5d_up = self.upsample(hx_5d)

        hx_4d = self.re_conv_4(torch.cat((hx_5d_up, hx_4), 1))
        hx_4d_up = self.upsample(hx_4d)

        hx_3d = self.re_conv_3(torch.cat((hx_4d_up, hx_3), 1))
        hx_3d_up = self.upsample(hx_3d)

        hx_2d = self.re_conv_2(torch.cat((hx_3d_up, hx_2), 1))
        hx_2d_up = self.upsample(hx_2d)

        hx_1d = self.re_conv_1(torch.cat((hx_2d_up, hx_1), 1))

        return hx_1d + hx_in


class RSU6(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(RSU6, self).__init__()

        self.conv_0 = CBRConv_3d(in_channel, out_channel, dirate=1)
        self.conv_1 = CBRConv_3d(out_channel, mid_channel, dirate=1)
        self.conv_2 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_3 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_4 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_5 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_6 = CBRConv_2d(mid_channel, mid_channel, dirate=1)

        self.re_conv_5 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_4 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_3 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_2 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_1 = CBRConv_3d(mid_channel * 2, out_channel, dirate=1)

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

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
        hx_6 = self.conv_6(hx_5)

        hx_5d = self.re_conv_5(torch.cat((hx_6, hx_5), 1))
        hx_5d_up = self.upsample(hx_5d)

        hx_4d = self.re_conv_4(torch.cat((hx_5d_up, hx_4), 1))
        hx_4d_up = self.upsample(hx_4d)

        hx_3d = self.re_conv_3(torch.cat((hx_4d_up, hx_3), 1))
        hx_3d_up = self.upsample(hx_3d)

        hx_2d = self.re_conv_2(torch.cat((hx_3d_up, hx_2), 1))
        hx_2d_up = self.upsample(hx_2d)

        hx_1d = self.re_conv_1(torch.cat((hx_2d_up, hx_1), 1))

        return hx_1d + hx_in


class RSU5(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(RSU5, self).__init__()

        self.conv_0 = CBRConv_3d(in_channel, out_channel, dirate=1)

        self.conv_1 = CBRConv_3d(out_channel, mid_channel, dirate=1)
        self.conv_2 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_3 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_4 = CBRConv_2d(mid_channel, mid_channel, dirate=1)
        self.conv_5 = CBRConv_2d(mid_channel, mid_channel, dirate=1)

        self.re_conv_4 = CBRConv_2d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_3 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_2 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_1 = CBRConv_3d(mid_channel * 2, out_channel, dirate=1)

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

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
        hx_5 = self.conv_5(hx_4)

        hx_4d = self.re_conv_4(torch.cat((hx_5, hx_4), 1))
        hx_4d_up = self.upsample(hx_4d)

        hx_3d = self.re_conv_3(torch.cat((hx_4d_up, hx_3), 1))
        hx_3d_up = self.upsample(hx_3d)

        hx_2d = self.re_conv_2(torch.cat((hx_3d_up, hx_2), 1))
        hx_2d_up = self.upsample(hx_2d)

        hx_1d = self.re_conv_1(torch.cat((hx_2d_up, hx_1), 1))

        return hx_1d + hx_in


class RSU4(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(RSU4, self).__init__()

        self.conv_0 = CBRConv_3d(in_channel, out_channel, dirate=1)

        self.conv_1 = CBRConv_3d(out_channel, mid_channel, dirate=1)
        self.conv_2 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_3 = CBRConv_3d(mid_channel, mid_channel, dirate=1)
        self.conv_4 = CBRConv_2d(mid_channel, mid_channel, dirate=1)

        self.re_conv_3 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_2 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=1)
        self.re_conv_1 = CBRConv_3d(mid_channel * 2, out_channel, dirate=1)

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

    def forward(self, x):
        hx = x
        hx_in = self.conv_0(hx)
        hx_1 = self.conv_1(hx_in)
        hx = self.pool(hx_1)

        hx_2 = self.conv_2(hx)
        hx = self.pool(hx_2)

        hx_3 = self.conv_3(hx)
        hx_4 = self.conv_4(hx_3)

        hx_3d = self.re_conv_3(torch.cat((hx_4, hx_3), 1))
        hx_3d_up = self.upsample(hx_3d)

        hx_2d = self.re_conv_2(torch.cat((hx_3d_up, hx_2), 1))
        hx_2d_up = self.upsample(hx_2d)

        hx_1d = self.re_conv_1(torch.cat((hx_2d_up, hx_1), 1))

        return hx_1d + hx_in


class RSU4F(nn.Module):

    def __init__(self, in_channel=3, mid_channel=16, out_channel=3):
        super(RSU4F, self).__init__()

        self.conv_0 = CBRConv_3d(in_channel, out_channel, dirate=1)
        self.conv_1 = CBRConv_3d(out_channel, mid_channel, dirate=1)
        self.conv_2 = CBRConv_3d(mid_channel, mid_channel, dirate=2)
        self.conv_3 = CBRConv_3d(mid_channel, mid_channel, dirate=4)
        self.conv_4 = CBRConv_3d(mid_channel, mid_channel, dirate=8)

        self.re_conv_3 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=4)
        self.re_conv_2 = CBRConv_3d(mid_channel * 2, mid_channel, dirate=2)
        self.re_conv_1 = CBRConv_3d(mid_channel * 2, out_channel, dirate=1)

    def forward(self, x):
        hx = x
        hx_in = self.conv_0(hx)
        hx_1 = self.conv_1(hx_in)
        hx_2 = self.conv_2(hx_1)
        hx_3 = self.conv_3(hx_2)
        hx_4 = self.conv_4(hx_3)

        hx_3d = self.re_conv_3(torch.cat((hx_4, hx_3), 1))
        hx_2d = self.re_conv_2(torch.cat((hx_3d, hx_2), 1))
        hx_1d = self.re_conv_1(torch.cat((hx_2d, hx_1), 1))

        return hx_1d + hx_in


class U2NET(nn.Module):

    def __init__(self, in_ch=1, out_ch=3):
        super(U2NET, self).__init__()

        channel = 16
        self.stage1 = RSU7(in_ch, channel, channel*2)
        self.stage2 = RSU6(channel*2, channel, channel*4)
        self.stage3 = RSU5(channel*4, channel*2, channel*8)
        self.stage4 = RSU4(channel*8, channel*4, channel*16)
        self.stage5 = RSU4F(channel*16, channel*8, channel*16)
        self.stage6 = RSU4F(channel*16, channel*8, channel*16)

        # decoder
        self.stage5d = RSU4F(channel*32, channel*8, channel*16)
        self.stage4d = RSU4(channel*32, channel*4, channel*8)
        self.stage3d = RSU5(channel*16, channel*2, channel*4)
        self.stage2d = RSU6(channel*8, channel, channel*2)
        self.stage1d = RSU7(channel*4, 16, channel*2)

        self.side1 = nn.Conv3d(channel*2, out_ch, 3, padding=1)
        self.side2 = nn.Conv3d(channel*2, out_ch, 3, padding=1)
        self.side3 = nn.Conv3d(channel*4, out_ch, 3, padding=1)
        self.side4 = nn.Conv3d(channel*8, out_ch, 3, padding=1)
        self.side5 = nn.Conv3d(channel*16, out_ch, 3, padding=1)
        self.side6 = nn.Conv3d(channel*16, out_ch, 3, padding=1)

        self.outconv = nn.Conv3d(6 * out_ch, out_ch, 1)

        self.pool = nn.MaxPool3d((2, 2, 2))
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.upsample_2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        self.upsample_3 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')
        self.upsample_4 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear')
        self.upsample_5 = nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear')
        self.upsample_6 = nn.Upsample(scale_factor=(32, 32, 32), mode='trilinear')

    def forward(self, x):
        hx = x

        hx1 = self.stage1(hx)
        hx = self.pool(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool(hx5)

        hx6 = self.stage6(hx)
        hx6up = self.upsample(hx6)

        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = self.upsample(hx5d)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = self.upsample(hx4d)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = self.upsample(hx3d)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = self.upsample(hx2d)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = self.upsample_2(d2)

        # print(hx3d.shape)
        d3 = self.side3(hx3d)
        d3 = self.upsample_3(d3)

        # print(hx4d.shape)
        d4 = self.side4(hx4d)
        d4 = self.upsample_4(d4)

        d5 = self.side5(hx5d)
        d5 = self.upsample_5(d5)

        d6 = self.side6(hx6)
        d6 = self.upsample_6(d6)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        # return F.sigmoid(d0)
        return F.sigmoid(d0), [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)]
