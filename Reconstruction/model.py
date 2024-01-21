import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import CT_denoise.denoising_V6.block as B
# from models.block_2D import decoding_block, encoding_block


class Getgradientnopadding(nn.Module):
    def __init__(self):
        super(Getgradientnopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


class FRB(nn.Module):
    def __init__(self, nc_in=32, nc_out=32):
        super(FRB, self).__init__()

        self.inter = nn.Conv3d(in_channels=nc_in, out_channels=nc_in,
                               kernel_size=(1, 1, 3), stride=(1, 1, 1),
                               padding=(0, 0, 1))

        self.intra = nn.Conv3d(in_channels=nc_in, out_channels=nc_in,
                               kernel_size=(5, 5, 1), stride=(1, 1, 1),
                               padding=(2, 2, 0))

        self.conv_1 = nn.Conv3d(in_channels=2 * nc_in, out_channels=nc_in,
                                kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))

        self.conv_2 = nn.Conv3d(in_channels=nc_in, out_channels=nc_out,
                                kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                padding=(0, 0, 0))

        self.norm = nn.LayerNorm([2 * nc_in, 512, 512, 5])
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x_inter = self.inter(x)
        x_intra = self.intra(x)

        x_fusion = torch.cat([x_intra, x_inter], dim=1)
        x = self.norm(x_fusion)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        #########
        return x


class RRDB(nn.Module):
    def __init__(self, nc, kernel_size=3):
        super(RRDB, self).__init__()
        self.RDB1 = nn.Conv3d(in_channels=nc, out_channels=nc,
                              kernel_size=(kernel_size, kernel_size, 1),
                              stride=1, padding=(1, 1, 0), bias=True)
        self.RDB2 = nn.Conv3d(in_channels=nc, out_channels=nc,
                              kernel_size=(kernel_size, kernel_size, 1),
                              stride=1, padding=(1, 1, 0), bias=True)
        self.RDB3 = nn.Conv3d(in_channels=nc, out_channels=nc,
                              kernel_size=(kernel_size, kernel_size, 1),
                              stride=1, padding=(1, 1, 0), bias=True)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x


class I2SR(nn.Module):
    def __init__(self, nc_in=1, nc_out=1, nc=16):
        super(I2SR, self).__init__()

        self.fea_conv = nn.Conv3d(in_channels=nc_in, out_channels=nc,
                                  kernel_size=3, stride=1, padding=1, bias=True)
        self.rb_blocks = nn.Sequential(
            RRDB(nc, kernel_size=3),
            RRDB(nc, kernel_size=3))
        self.fusion_block = nn.Sequential(
            FRB(nc_in=nc, nc_out=nc),
            FRB(nc_in=nc, nc_out=nc),
            FRB(nc_in=nc, nc_out=2 * nc),
            FRB(nc_in=2 * nc, nc_out=nc),
            FRB(nc_in=nc, nc_out=nc),
            FRB(nc_in=nc, nc_out=nc))

        self.get_g_nopadding = Getgradientnopadding()

        self.final_conv = nn.Sequential(
            nn.Conv3d(in_channels=nc, out_channels=nc,
                      kernel_size=(3, 3, 1), stride=1,
                      padding=(1, 1, 0), bias=True),
            nn.Conv3d(in_channels=nc, out_channels=nc_out,
                      kernel_size=(3, 3, 1), stride=1,
                      padding=(1, 1, 0), bias=True))

    def forward(self, x0):
        x = x0
        # x_grad = x0.clone()
        # for i in range(x_grad.shape[-1]):
        #     x_grad[:, :, :, :, i] = self.get_g_nopadding(x0[:, :, :, :, i])

        # x = torch.concat([x0, x_grad], dim=1)
        x = self.fea_conv(x)
        x = self.rb_blocks(x)
        x = self.fusion_block(x)
        # print(x.shape)

        x_out = self.final_conv(x)
        # print(x.shape)
        for i in range(x_out.shape[-1]):
            x_out[:, :, :, :, i] += x0[:, :, :, :, 2]

        return x_out

class Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernal=3, padding=1):
        super(Residual_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                      kernel_size=(kernal, kernal), stride=(1, 1), padding=padding, bias=False),
            nn.InstanceNorm2d(self.out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel,
                      kernel_size=(kernal, kernal), stride=(1, 1), padding=padding, bias=False),
            nn.InstanceNorm2d(self.out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        output = x + self.conv2(x)
        return output

class Full_CNN(nn.Module):
    def __init__(self, inchannels=3, outchannels=5):
        super(Full_CNN, self).__init__()

        self.in_channels = inchannels
        self.out_channels = outchannels

        self.layer_0_left = Residual_Block(self.in_channels, 16)
        self.layer_0_right = Residual_Block(self.in_channels, 16)

        self.layer_1_left = Residual_Block(16, 32)
        self.layer_1_right = Residual_Block(16, 32)

        self.layer_2_left = Residual_Block(32, 64)
        self.layer_2_right = Residual_Block(32, 64)

        self.layer_3_left = Residual_Block(64, 128)
        self.layer_3_right = Residual_Block(64, 128)

        self.layer_4_left = Residual_Block(128, 256)
        self.layer_4_right = Residual_Block(128, 256)

        self.layer_5_left = Residual_Block(256, 128)
        self.layer_5_right = Residual_Block(256, 128)

        self.layer_6_left = Residual_Block(128, 64)
        self.layer_6_right = Residual_Block(128, 64)

        self.layer_7_left = Residual_Block(64, 32)
        self.layer_7_right = Residual_Block(64, 32)

        self.layer_8_left = Residual_Block(32, 16)
        self.layer_8_right = Residual_Block(32, 16)

        self.upscale = Residual_Block(32, 16)

        self.upsample = nn.Conv2d(in_channels=16, out_channels=self.out_channels, kernel_size=(3, 3),
                                  stride=(1, 1), padding=1, bias=True)

    def forward(self, left, right):
        left = self.layer_0_left(left)
        right = self.layer_0_right(right)

        left = self.layer_1_left(left)
        right = self.layer_1_right(right)

        left = self.layer_2_left(left)
        right = self.layer_2_right(right)

        left = self.layer_3_left(left)
        right = self.layer_3_right(right)

        left = self.layer_4_left(left)
        right = self.layer_4_right(right)

        left = self.layer_5_left(left)
        right = self.layer_5_right(right)

        left = self.layer_6_left(left)
        right = self.layer_6_right(right)

        left = self.layer_7_left(left)
        right = self.layer_7_right(right)

        # left, right = self.IAM_2(left, right)

        left = self.layer_8_left(left)
        right = self.layer_8_right(right)

        x = torch.cat([left, right], dim=1)
        x_1 = self.upscale(x)
        out = self.upsample(x_1)
        # print(x.shape)
        return out + left[:, 2:3]