import torch
import torch.nn as nn
from Reconstruction.layers import Residual_Block, IAM


class Generator(nn.Module):
    def __init__(self, inchannels=3, outchannels=5):
        super(Generator, self).__init__()

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

        self.IAM_1 = IAM(channels=32)
        self.IAM_2 = IAM(channels=32)

        self.upscale = Residual_Block(32, 16)

        self.upsample = nn.Conv2d(in_channels=16, out_channels=self.out_channels, kernel_size=(3, 3),
                                  stride=(1, 1), padding=1, bias=True)

    def forward(self, left, right):
        left = self.layer_0_left(left)
        right = self.layer_0_right(right)

        left = self.layer_1_left(left)
        right = self.layer_1_right(right)

        # left, right = self.IAM_1(left, right)

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
        return out