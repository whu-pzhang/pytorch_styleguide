import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["UNet"]


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


def conv3x3(in_channels, out_channels, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=False)


def upconv2x2(in_channels, out_channels):
    return nn.Sequential(
        conv1x1(in_channels, out_channels),
        Upsample(scale_factor=2)
    )


def double_conv3x3(in_channels, out_channels):
    """(conv->BN->ReLU) * 2"""
    return nn.Sequential(
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        conv3x3(out_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Upsample(nn.Module):
    """
    Custom Upsample layer (nn.Upsample give deprecated warning message)
    """

    def __init__(self, scale_factor=2, mode="bilinear"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = double_conv3x3(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.upconv = upconv2x2(in_channels, out_channels)
        self.conv = double_conv3x3(2 * out_channels, out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat((from_down, from_up), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, start_filters=64, depth=5, up_mode="transpose", merge_mode="cat"):
        super(UNet, self).__init__()

        self.down1 = DownConv(in_channels, start_filters)
        self.down2 = DownConv(start_filters, start_filters * 2)
        self.down3 = DownConv(start_filters * 2, start_filters * 4)
        self.down4 = DownConv(start_filters * 4, start_filters * 8)
        self.down5 = DownConv(start_filters * 8, start_filters * 16)

        self.maxpool = nn.MaxPool2d(2)

        self.up1 = UpConv(start_filters * 16, start_filters * 8)
        self.up2 = UpConv(start_filters * 8, start_filters * 4)
        self.up3 = UpConv(start_filters * 4, start_filters * 2)
        self.up4 = UpConv(start_filters * 2, start_filters)

        self.last_conv = conv1x1(start_filters, num_classes)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        conv1 = self.down1(x)
        x = self.maxpool(conv1)
        conv2 = self.down2(x)
        x = self.maxpool(conv2)
        conv3 = self.down3(x)
        x = self.maxpool(conv3)
        conv4 = self.down4(x)
        x = self.maxpool(conv4)
        conv5 = self.down5(x)

        x = self.up1(conv4, conv5)
        x = self.up2(conv3, x)
        x = self.up3(conv2, x)
        x = self.up4(conv1, x)

        x = self.last_conv(x)
        return x


# if __name__ == "__main__":
#     model = UNet(in_channels=3, num_classes=10, up_mode="transpose")
#     from torchsummary import summary
#     summary(model, (3, 256, 256))
