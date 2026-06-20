import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels,
                               kernel_size=3, padding=1, groups=in_channels)
        self.conv2 = nn.Conv3d(in_channels, mid_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(mid_channels, in_channels, kernel_size=1)
        self.conv4 = nn.Conv3d(2 * in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.conv3(self.act(self.conv2(self.norm(self.conv1(x))))))
        x = self.conv4(torch.cat([residual, x], dim=1))
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels,
                               kernel_size=3, stride=2, padding=1, groups=in_channels)
        self.conv2 = nn.Conv3d(in_channels, mid_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(mid_channels, in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.act = nn.GELU()
        self.res = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=2)

    def forward(self, x):
        residual = self.res(x)
        x = self.act(self.conv3(self.act(self.conv2(self.norm(self.conv1(x))))))
        return torch.cat([residual, x], dim=1)


class UpSample(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, in_channels,
                                        kernel_size=2, stride=2, groups=in_channels)
        self.conv2 = nn.Conv3d(in_channels, mid_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(mid_channels, in_channels // 2, kernel_size=1)
        self.conv4 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.act = nn.GELU()
        self.res = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        residual = self.res(x)
        x = self.act(self.conv3(self.act(self.conv2(self.norm(self.conv1(x))))))
        return self.conv4(torch.cat([residual, x], dim=1))


class HiPaSNet(nn.Module):
    def __init__(self, in_channels, mid_channels=16, out_channels=2, r=4):
        super().__init__()

        self.pconv = nn.Conv3d(2, 2, kernel_size=3, padding=1)
        self.trans = nn.Sequential(
            nn.Conv3d(in_channels + 2, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
        )

        self.stem = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)

        c = mid_channels
        self.enc_block_0 = ResBlock(c, r * c, c)
        self.down_0 = DownSample(c, r * c)

        self.enc_block_1 = ResBlock(2 * c, r * 2 * c, 2 * c)
        self.down_1 = DownSample(2 * c, r * 2 * c)

        self.enc_block_2 = ResBlock(4 * c, r * 4 * c, 4 * c)
        self.down_2 = DownSample(4 * c, r * 4 * c)

        self.enc_block_3 = ResBlock(8 * c, r * 8 * c, 8 * c)
        self.down_3 = DownSample(8 * c, r * 8 * c)

        self.bottleneck = nn.Conv3d(16 * c, 16 * c, kernel_size=1)

        self.up_3 = UpSample(16 * c, r * 16 * c)
        self.dec_block_3 = ResBlock(16 * c, r * 8 * c, 8 * c)

        self.up_2 = UpSample(8 * c, r * 8 * c)
        self.dec_block_2 = ResBlock(8 * c, r * 4 * c, 4 * c)

        self.up_1 = UpSample(4 * c, r * 4 * c)
        self.dec_block_1 = ResBlock(4 * c, r * 2 * c, 2 * c)

        self.up_0 = UpSample(2 * c, r * 2 * c)
        self.dec_block_0 = ResBlock(2 * c, r * c, c)

        self.out = nn.Sequential(
            nn.Conv3d(c, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, p=None):
        if p is not None:
            x = self.trans(torch.cat([x, self.pconv(p)], dim=1))

        x = self.stem(x)

        e0 = self.enc_block_0(x)
        e1 = self.enc_block_1(self.down_0(e0))
        e2 = self.enc_block_2(self.down_1(e1))
        e3 = self.enc_block_3(self.down_2(e2))

        x = self.bottleneck(self.down_3(e3))

        x = self.dec_block_3(torch.cat([self.up_3(x), e3], dim=1))
        x = self.dec_block_2(torch.cat([self.up_2(x), e2], dim=1))
        x = self.dec_block_1(torch.cat([self.up_1(x), e1], dim=1))
        x = self.dec_block_0(torch.cat([self.up_0(x), e0], dim=1))

        return self.out(x)

