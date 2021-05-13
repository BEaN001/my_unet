from .resnet import *


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, bridge_chans, out_chans, up_mode):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
          self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
          self.up = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_chans, out_chans, kernel_size=1),
          )
        self.conv_block = BasicBlock(out_chans + bridge_chans, out_chans)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([crop, up], dim=1)
        out = self.conv_block(out)
        return out


class ResNet_UNet(nn.Module):
    """
    resnet_unet
    """

    def __init__(self, in_chans=1, n_classes=2, up_mode='upconv'):
        super(ResNet_UNet, self).__init__()
        self.n_classes = n_classes
        self.up_mode = up_mode
        assert self.up_mode in ('upconv', 'upsample')

        self.encoder = resnet34(in_chans)
        in_chans = 512 * self.encoder.block.expansion

        self.decoder = nn.ModuleList()
        for i in range(3):
            self.decoder.append(UNetUpBlock(in_chans, in_chans // 2, in_chans // 2, self.up_mode))
            in_chans //= 2
        self.decoder.append(UNetUpBlock(in_chans, 64, 64, self.up_mode))
        self.up_final = UNetUpBlock(64, 3, 64, self.up_mode)  # add up sample final layer

        self.cls_conv = nn.Conv2d(64, self.n_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoding
        f0 = x
        f1, f2, f3, f4, f5, out = self.encoder(x)
        bridges = [f1, f2, f3, f4]
        x = f5

        # decoding
        for i, decode_layer in enumerate(self.decoder):
            x = decode_layer(x, bridges[-i - 1])
        x = self.up_final(x, f0)  # add upsample final layer
        score = self.cls_conv(x)
        return score


if __name__ == "__main__":
    x = torch.randn((2, 3, 224, 224), dtype=torch.float32)
    unet = ResNet_UNet(in_chans=3, n_classes=2, up_mode='upconv')
    print(x.shape)
    y = unet(x)
    print(y.shape)
    from torchsummary import summary
    summary_vision = summary(unet, (3, 224, 224))
