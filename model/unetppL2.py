import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans, stride):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(x)))
        return out


class UpConvBlock(nn.Module):
    def __init__(self, in_chans, bridge_chans_list, out_chans):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_chans + sum(bridge_chans_list), out_chans, 1)

    def forward(self, x, bridge_list):
        x = self.up(x)
        x = torch.cat([x] + bridge_list, dim=1)
        out = self.conv_block(x)
        return out


class UNetpp(nn.Module):
    def __init__(self, in_chans=1, n_classes=2):
        super(UNetpp, self).__init__()

        feat_chans = [64, 128, 256]
        self.conv_x00 = ConvBlock(in_chans, feat_chans[0], 1)
        self.conv_x10 = ConvBlock(feat_chans[0], feat_chans[1], 2)
        self.conv_x20 = ConvBlock(feat_chans[1], feat_chans[2], 2)

        self.conv_x01 = UpConvBlock(feat_chans[1], [feat_chans[0]], feat_chans[0])
        self.conv_x11 = UpConvBlock(feat_chans[2], [feat_chans[1]], feat_chans[1])
        self.conv_x02 = UpConvBlock(feat_chans[1], [feat_chans[0], feat_chans[0]], feat_chans[0])

        self.cls_conv_x01 = nn.Conv2d(feat_chans[0], n_classes, kernel_size=1)
        self.cls_conv_x02 = nn.Conv2d(feat_chans[0], n_classes, kernel_size=1)

    def forward(self, x):
        x00 = self.conv_x00(x)
        x10 = self.conv_x10(x00)
        x20 = self.conv_x20(x10)
        x01 = self.conv_x01(x10, [x00])
        x11 = self.conv_x11(x20, [x10])
        x02 = self.conv_x02(x11, [x00, x01])
        out01 = self.cls_conv_x01(x01)
        out02 = self.cls_conv_x02(x02)

        print('x00', x00.shape)
        print('x10', x10.shape)
        print('x20', x20.shape)
        print('x01', x01.shape)
        print('x11', x11.shape)
        print('x02', x02.shape)
        print('out01', out01.shape)
        print('out02', out02.shape)

        return out01, out02


if __name__ == "__main__":
    x = torch.randn((2, 3, 224, 224), dtype=torch.float32)
    model = UNetpp(in_chans=3, n_classes=3)
    y1, y2 = model(x)
    from torchsummary import summary
    summary_vision = summary(model, (3, 224, 224))