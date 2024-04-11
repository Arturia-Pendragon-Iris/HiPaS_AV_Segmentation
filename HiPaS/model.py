import torch.nn as nn
import torch
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels,
                               kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=in_channels)
        self.conv2 = nn.Conv3d(in_channels, mid_channels,
                               kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv3 = nn.Conv3d(mid_channels, in_channels,
                               kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv4 = nn.Conv3d(2 * in_channels, out_channels,
                               kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.norm = nn.GroupNorm(in_channels, in_channels, eps=1e-05, affine=True)
        self.act = nn.GELU(approximate='none')

    def forward(self, x):
        # encoder
        residual_1 = x
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        x = torch.concatenate([residual_1, x], dim=1)
        x = self.act(x)
        x = self.conv4(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels,
                               kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), groups=in_channels)
        self.conv2 = nn.Conv3d(in_channels, mid_channels,
                               kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv3 = nn.Conv3d(mid_channels, in_channels,
                               kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.norm = nn.GroupNorm(in_channels, in_channels, eps=1e-05, affine=True)
        self.act = nn.GELU(approximate='none')
        self.res = nn.Conv3d(in_channels, in_channels,
                             kernel_size=(1, 1, 1), stride=(2, 2, 2))

    def forward(self, x):
        # encoder
        residual_1 = self.res(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        x = torch.concatenate([residual_1, x], dim=1)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, in_channels,
                                        kernel_size=(2, 2, 2), stride=(2, 2, 2), groups=in_channels)
        self.conv2 = nn.Conv3d(in_channels, mid_channels,
                               kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv3 = nn.Conv3d(mid_channels, in_channels // 2,
                               kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels, in_channels // 2,
                               kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.norm = nn.GroupNorm(in_channels, in_channels, eps=1e-05, affine=True)
        self.act = nn.GELU(approximate='none')
        self.res = nn.ConvTranspose3d(in_channels, in_channels // 2,
                                      kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        # encoder
        residual_1 = self.res(x)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        x = torch.concatenate([residual_1, x], dim=1)
        x = self.conv4(x)
        return x


class HiPaSNet(nn.Module):
    def __init__(self, in_channels, mid_channels=16, out_channels=2, r=4):
        super().__init__()

        self.pconv = nn.Conv3d(2, 2,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.trans = nn.Sequential(
            nn.Conv3d(in_channels + 2, in_channels,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.GELU(approximate='none'),
            nn.Conv3d(in_channels, in_channels,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )

        self.stem = nn.Conv3d(in_channels, mid_channels,
                              kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.enc_block_0 = ResBlock(mid_channels, r * mid_channels, mid_channels)
        self.down_0 = DownSample(mid_channels, r * mid_channels)

        self.enc_block_1 = ResBlock(2 * mid_channels, r * 2 * mid_channels, 2 * mid_channels)
        self.down_1 = DownSample(2 * mid_channels, r * 2 * mid_channels)

        self.enc_block_2 = ResBlock(4 * mid_channels, r * 4 * mid_channels, 4 * mid_channels)
        self.down_2 = DownSample(4 * mid_channels, r * 4 * mid_channels)

        self.enc_block_3 = ResBlock(8 * mid_channels, r * 8 * mid_channels, 8 * mid_channels)
        self.down_3 = DownSample(8 * mid_channels, r * 8 * mid_channels)

        self.bottom_neck= nn.Conv3d(16 * mid_channels, 16 * mid_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        self.up_3 = UpSample(16 * mid_channels, r * 16 * mid_channels)
        self.dec_block_3 = ResBlock(16 * mid_channels, r * 8 * mid_channels, 8 * mid_channels)

        self.up_2 = UpSample(8 * mid_channels, r * 8 * mid_channels)
        self.dec_block_2 = ResBlock(8 * mid_channels, r * 4 * mid_channels, 4 * mid_channels)

        self.up_1 = UpSample(4 * mid_channels, r * 4 * mid_channels)
        self.dec_block_1 = ResBlock(4 * mid_channels, r * 2 * mid_channels, 2 * mid_channels)

        self.up_0 = UpSample(2 * mid_channels, r * 2 * mid_channels)
        self.dec_block_0 = ResBlock(2 * mid_channels, r * mid_channels, mid_channels)

        self.out = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x, p):
        if not p is None:
            x = self.trans(torch.concatenate((x, self.pconv(p)), dim=1))
        # encoder
        x = self.stem(x)
        residual_0 = self.enc_block_0(x)
        x = self.down_0(residual_0)

        residual_1 = self.enc_block_1(x)
        x = self.down_1(residual_1)

        residual_2 = self.enc_block_2(x)
        x = self.down_2(residual_2)

        residual_3 = self.enc_block_3(x)
        x = self.down_3(residual_3)

        x = self.bottom_neck(x)

        x = self.up_3(x)
        x = torch.concatenate([x, residual_3], dim=1)
        x = self.dec_block_3(x)
        print(x.shape)

        x = self.up_2(x)
        x = torch.concatenate([x, residual_2], dim=1)
        x = self.dec_block_2(x)

        x = self.up_1(x)
        x = torch.concatenate([x, residual_1], dim=1)
        x = self.dec_block_1(x)

        x = self.up_0(x)
        x = torch.concatenate([x, residual_0], dim=1)
        x = self.dec_block_0(x)

        x = self.out(x)

        return x

