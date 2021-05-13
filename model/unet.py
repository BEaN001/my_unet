import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class NaiveUNet(nn.Module):
    def __init__(self, in_channel=1, n_class=2):
        super(NaiveUNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(in_channel, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2
        )

        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )

        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )

        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )

        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=n_class,
            kernel_size=1
        )

    def forward(self, image):
        # bs, c, h, w
        # encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        # print(x9.shape)

        # decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], dim=1))

        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], dim=1))

        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], dim=1))

        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], dim=1))

        x = self.out(x)
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, padding):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = UNetConvBlock(in_chans, out_chans, padding, True)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([crop1, up], dim=1)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):

    def __init__(self, in_chans=1, n_classes=2, padding=False, up_mode='upconv'):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.padding = padding
        self.up_mode = 'upconv'
        assert self.up_mode in ('upconv', 'upsample')

        out_chans = 64
        self.encoder = nn.ModuleList()
        for i in range(5):
            self.encoder.append(UNetConvBlock(in_chans, out_chans, self.padding, batch_norm=False))
            in_chans = out_chans
            out_chans *= 2

        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(UNetUpBlock(in_chans, in_chans // 2, self.up_mode, self.padding))
            in_chans //= 2

        self.cls_conv = nn.Conv2d(in_chans, self.n_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoding
        bridges = []
        # print("encoder")
        for i, encode_layer in enumerate(self.encoder):
            x = encode_layer(x)
            if i < len(self.encoder) - 1:
                bridges.append(x)
                x = F.max_pool2d(x, kernel_size=2)
            # print(x.shape)

        # decoding
        # print("decoder")
        for i, decode_layer in enumerate(self.decoder):
            x = decode_layer(x, bridges[-i - 1])
            # print(x.shape)

        score = self.cls_conv(x)
        return score


# Create the U-net model
# Let's create a smal u-net like model as a toy example.

# Define the convolutional block
# First we define the convolutional block so we do not need to re-write it every time
class conv_block(nn.Module):
    """
    Define the convolutional - batch norm - relu block to avoid re-writing it
    every time
    """

    def __init__(self, in_size, out_size, kernel_size=3, padding=1, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# Define the network
# Then we define the network keeping in mind that down_1 output size should match with
# the scaled output of up_2, while down_2 size should match with the scaled output of the middle
# convolution of the u-net model.
class small_UNET_256(nn.Module):
    """
    Define UNET model that accepts a 256 input and mostly uses 3x3 kernels
    with stride and padding = 1. It reduces the size of the image to 8x8 pixels
    ** It might not work if the input 'x' is not a square.
    """

    def __init__(self):
        super(small_UNET_256, self).__init__()

        self.down_1 = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 32, stride=2, padding=1))

        self.down_2 = nn.Sequential(
            conv_block(32, 64),
            conv_block(64, 128))

        self.middle = conv_block(128, 128, kernel_size=1, padding=0)

        self.up_2 = nn.Sequential(
            conv_block(256, 128),
            conv_block(128, 32))

        self.up_1 = nn.Sequential(
            conv_block(64, 64),
            conv_block(64, 32))

        self.output = nn.Sequential(
            conv_block(32, 16),
            conv_block(16, 1, kernel_size=1, padding=0))

    def forward(self, x):
        down1 = self.down_1(x)
        out = F.max_pool2d(down1, kernel_size=2, stride=2)

        down2 = self.down_2(out)
        out = F.max_pool2d(down2, kernel_size=2, stride=2)

        out = self.middle(out)

        out = F.upsample(out, scale_factor=2)
        out = torch.cat([down2, out], 1)
        out = self.up_2(out)

        out = F.upsample(out, scale_factor=2)
        out = torch.cat([down1, out], 1)
        out = self.up_1(out)

        out = F.upsample(out, scale_factor=2)
        return self.output(out)


if __name__ == "__main__":
    # simpleUNet
    print(30 * '*')
    print("small_UNET_256: input: b*3*256*256; output: b*1*256*256")
    image = torch.rand((1, 3, 256, 256))
    model = small_UNET_256()
    print(model(image).shape)

    # NaiveUNet scrath from paper
    print(30*'*')
    print("NaiveUNet: input: b*3*572*572; output: b*class*388*388")
    image = torch.rand((1, 3, 572, 572))
    model = NaiveUNet(in_channel=3, n_class=3)
    print(model(image).shape)

    # UNet
    print(30 * '*')
    print("UNet: input: b*3*572*572; output: b*class*388*388")
    x = torch.randn((1, 3, 572, 572), dtype=torch.float32)
    unet = UNet(in_chans=3, padding=False, n_classes=3)
    # print(x.shape)
    y = unet(x)
    print(y.shape)
    from torchsummary import summary
    summary_vision = summary(unet, (3, 572, 572))

    # # UNet
    # print(30 * '*')
    # print("UNet: input: b*3*572*572; output: b*class*388*388")
    # x = torch.randn((2, 1, 224, 224), dtype=torch.float32)
    # unet = UNet(in_chans=1, padding=False, n_classes=2, up_mode='upconv')
    # # print(x.shape)
    # y = unet(x)
    # print(y.shape)
    # from torchsummary import summary
    #
    # summary_vision = summary(unet, (1, 224, 224))
