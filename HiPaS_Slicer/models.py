import torch.nn as nn
import torch
import torch.nn.functional as F


class SimpleBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 7,
                 do_res: int = True,
                 n_groups: int or None = None,
                 ):
        super().__init__()

        self.do_res = do_res

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        self.norm = nn.GroupNorm(
            num_groups=in_channels,
            num_channels=in_channels
        )

        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.act = nn.GELU()

        self.conv3 = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class DownSample(SimpleBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False)

        self.res_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2
        )

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        res = self.res_conv(x)
        x1 = x1 + res

        return x1


class UpSample(SimpleBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, )

        self.res_conv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2
        )

        self.conv1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        x1 = super().forward(x)
        x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        res = self.res_conv(x)

        res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
        x1 = x1 + res

        return x1


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.conv_out = nn.Sequential(nn.ConvTranspose3d(in_channels, n_classes, kernel_size=1))

    def forward(self, x):
        return self.conv_out(x)


class UNet(nn.Module):
    """
    Main UNet architecture
    """

    def __init__(self, in_channel=1, num_classes=1, active="none", channel=16):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel, channel, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel))
        self.maxpool1 = nn.Conv3d(channel, channel, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv3d(channel, channel * 2, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel * 2))
        self.maxpool2 = nn.Conv3d(channel * 2, channel * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Sequential(
            nn.Conv3d(channel * 2, channel * 4, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel * 4),
            nn.Conv3d(channel * 4, channel * 4, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel * 4))
        self.maxpool3 = nn.Conv3d(channel * 4, channel * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4 = nn.Sequential(
            nn.Conv3d(channel * 4, channel * 8, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel * 8),
            nn.Conv3d(channel * 8, channel * 8, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel * 8))
        self.maxpool4 = nn.Conv3d(channel * 8, channel * 8, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # center
        self.center = nn.Sequential(
            nn.Conv3d(channel * 8, channel * 16, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel * 16))

        # decoding
        self.decode4 = nn.Sequential(
            nn.Upsample(mode="trilinear", scale_factor=2, align_corners=True),
            nn.Conv3d(channel * 16, channel * 8, kernel_size=(1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(channel * 8, channel * 8, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel * 8))
        self.decode3 = nn.Sequential(
            nn.Upsample(mode="trilinear", scale_factor=2, align_corners=True),
            nn.Conv3d(channel * 16, channel * 8, kernel_size=(1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(channel * 8, channel * 4, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel * 4))
        self.decode2 = nn.Sequential(
            nn.Upsample(mode="trilinear", scale_factor=2, align_corners=True),
            nn.Conv3d(channel * 8, channel * 4, kernel_size=(1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(channel * 4, channel * 2, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel * 2))
        self.decode1 = nn.Sequential(
            nn.Upsample(mode="trilinear", scale_factor=2, align_corners=True),
            nn.Conv3d(channel * 4, channel * 2, kernel_size=(1, 1, 1)),
            nn.PReLU(),
            nn.Conv3d(channel * 2, channel * 1, kernel_size=3, padding=1, stride=1),
            nn.PReLU(),
            nn.BatchNorm3d(channel * 1))

        if active == "sigmoid":
            self.final = nn.Sequential(nn.Conv3d(channel * 2, num_classes, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                                 padding=(1, 1, 1)),
                                       nn.Sigmoid(dim=1))
        elif active == "softmax":
            self.final = nn.Sequential(nn.Conv3d(channel * 2, num_classes, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                                 padding=(1, 1, 1)),
                                       nn.Softmax(dim=1))
        else:
            self.final = nn.Sequential(nn.Conv3d(channel * 2, num_classes, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                                 padding=(1, 1, 1)))

    def forward(self, input, do_residual=False):

        if do_residual:
            residual = input
        # encoding
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        # print(conv2.shape)
        maxpool2 = self.maxpool2(conv2)
        # print(maxpool2.shape)

        conv3 = self.conv3(maxpool2)
        # print(conv3.shape)
        maxpool3 = self.maxpool3(conv3)
        # print(maxpool3.shape)

        conv4 = self.conv4(maxpool3)
        # print(conv4.shape)
        maxpool4 = self.maxpool4(conv4)
        # print(maxpool4.shape)

        # center
        center = self.center(maxpool4)
        # print(center.shape)

        # decoding
        decode4 = self.decode4(center)
        decode4 = torch.cat([decode4, conv4], 1)
        decode3 = self.decode3(decode4)
        decode3 = torch.cat([decode3, conv3], 1)
        decode2 = self.decode2(decode3)
        decode2 = torch.cat([decode2, conv2], 1)
        decode1 = self.decode1(decode2)
        decode1 = torch.cat([decode1, conv1], 1)

        final = self.final(decode1)

        if do_residual:
            final += residual

        return final


class MedNext(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 block_counts: list = (2, 2, 2, 2, 2, 2, 2, 2, 2),  # Can be used to test staging ratio:
                 dim='2d',  # 2d or 3d
                 ):

        super().__init__()

        self.do_ds = deep_supervision
        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        self.stem = nn.Conv3d(in_channels=in_channels,
                              out_channels=n_channels,
                              kernel_size=(1, 1, 1),
                              stride=(1, 1, 1))
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            SimpleBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[0])]
                                         )

        self.down_0 = DownSample(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
        )

        self.enc_block_1 = nn.Sequential(*[
            SimpleBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[1])]
                                         )

        self.down_1 = DownSample(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
        )

        self.enc_block_2 = nn.Sequential(*[
            SimpleBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[2])]
                                         )

        self.down_2 = DownSample(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
        )

        self.enc_block_3 = nn.Sequential(*[
            SimpleBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[3])]
                                         )

        self.down_3 = DownSample(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
        )

        self.bottleneck = nn.Sequential(*[
            SimpleBlock(
                in_channels=n_channels * 16,
                out_channels=n_channels * 16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[4])]
                                        )

        self.up_3 = UpSample(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
        )

        self.dec_block_3 = nn.Sequential(*[
            SimpleBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[5])]
                                         )

        self.up_2 = UpSample(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
        )

        self.dec_block_2 = nn.Sequential(*[
            SimpleBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[6])]
                                         )

        self.up_1 = UpSample(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
        )

        self.dec_block_1 = nn.Sequential(*[
            SimpleBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[7])]
                                         )

        self.up_0 = UpSample(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
        )

        self.dec_block_0 = nn.Sequential(*[
            SimpleBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
            )
            for i in range(block_counts[8])]
                                         )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes)

        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes)
            self.out_4 = OutBlock(in_channels=n_channels * 16, n_classes=n_classes)

        self.block_counts = block_counts

    def forward(self, x):
        x = self.stem(x)

        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)
        if self.do_ds:
            x_ds_4 = self.out_4(x)

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3
        x = self.dec_block_3(dec_x)

        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


class RED_CNN(nn.Module):
    def __init__(self, in_chans=1, out_ch=96, final_chans=1):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, final_chans, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out
