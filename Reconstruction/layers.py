import torch
import torch.nn as nn


class Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel,
                      kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.InstanceNorm2d(self.out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channel, out_channels=self.out_channel,
                      kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.InstanceNorm2d(self.out_channel, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        output = x + self.conv2(x)
        return output


class IAM(nn.Module):
    def __init__(self, channels):
        super(IAM, self).__init__()
        self.u1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.u2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)

        self.d1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.d2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)

        self.out_1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
        self.out_2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

        self.softmax = nn.Softmax(-1)

    def forward(self, u, d):  # B * C * H * W
        b, c, h, w = u.shape
        u1 = self.u1(u).permute(0, 2, 3, 1)
        u2 = self.u2(u).permute(0, 2, 1, 3)

        d1 = self.d1(d).permute(0, 2, 3, 1)
        d2 = self.d2(d).permute(0, 2, 1, 3)

        u_d = torch.bmm(u1.contiguous().view(-1, w, c),
                        d2.contiguous().view(-1, c, w))
        u_d = self.softmax(u_d)

        d_u = torch.bmm(d1.contiguous().view(-1, w, c),
                        u2.contiguous().view(-1, c, w))
        d_u = self.softmax(d_u)

        buffer_d = d.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_d = torch.bmm(u_d, buffer_d).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)

        buffer_u = u.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_u = torch.bmm(d_u, buffer_u).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W

        out_u = self.out_1(torch.concat((u, buffer_d), dim=1))
        out_d = self.out_2(torch.concat((d, buffer_u), dim=1))

        return out_u, out_d


class CNA3d(nn.Module):  # conv + norm + activation
    def __init__(self, in_channels, out_channels, bias=True, norm_args=None,
                 activation_args=None):
        super().__init__()
        self.norm_args = norm_args
        self.activation_args = activation_args

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kSize, stride=stride, padding=padding, bias=bias)

        if norm_args is not None:
            self.norm = nn.InstanceNorm3d(out_channels, **norm_args)

        if activation_args is not None:
            self.activation = nn.LeakyReLU(**activation_args)

    def forward(self, x):
        x = self.conv(x)

        if self.norm_args is not None:
            x = self.norm(x)

        if self.activation_args is not None:
            x = self.activation(x)
        return x


class CB3d(nn.Module):  # conv block 3d
    def __init__(self, in_channels, out_channels, kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1), bias=True,
                 norm_args: tuple = (None, None), activation_args: tuple = (None, None)):
        super().__init__()

        self.conv1 = CNA3d(in_channels, out_channels, kSize=kSize[0], stride=stride[0],
                           padding=padding, bias=bias, norm_args=norm_args[0], activation_args=activation_args[0])

        self.conv2 = CNA3d(out_channels, out_channels, kSize=kSize[1], stride=stride[1],
                           padding=padding, bias=bias, norm_args=norm_args[1], activation_args=activation_args[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x