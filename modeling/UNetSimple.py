from torch import nn
from torch.nn import functional as F
import torch
import torchvision


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x

class UNetSimple(nn.Module):

    def __init__(self, **kwargs):
        super(UNetSimple, self).__init__()

        self.num_classes = 4

        self.down1 = nn.Sequential(
            ConvBn2d(3,  64, kernel_size=3, stride=1, padding=1),
            ConvBn2d(64,  64, kernel_size=3, stride=1, padding=1),
        )
        self.down2 = nn.Sequential(
            ConvBn2d(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBn2d(128, 128, kernel_size=3, stride=1, padding=1),
        )
        self.down3 = nn.Sequential(
            ConvBn2d(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.down4 = nn.Sequential(
            ConvBn2d(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
        )
        self.down5 = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
        )

        self.same = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
        )

        self.up5 = nn.Sequential(
            ConvBn2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
        )

        self.up4 = nn.Sequential(
            ConvBn2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBn2d(512, 256, kernel_size=3, stride=1, padding=1),
        )
        self.up3 = nn.Sequential(
            ConvBn2d(512, 256, kernel_size=3, stride=1, padding=1),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBn2d(256, 128, kernel_size=3, stride=1, padding=1),
        )
        self.up2 = nn.Sequential(
            ConvBn2d(256, 128, kernel_size=3, stride=1, padding=1),
            ConvBn2d(128,  64, kernel_size=3, stride=1, padding=1),
        )
        self.up1 = nn.Sequential(
            ConvBn2d(128,  64, kernel_size=3, stride=1, padding=1),
            ConvBn2d(64,  64, kernel_size=3, stride=1, padding=1),
        )
        self.feature = nn.Sequential(
            ConvBn2d(64,  64, kernel_size=1, stride=1, padding=0),
        )
        self.logit = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        # input = torch.unsqueeze(input,1)
        down1 = self.down1(input)
        f = F.max_pool2d(down1, kernel_size=2, stride=2)
        down2 = self.down2(f)
        f = F.max_pool2d(down2, kernel_size=2, stride=2)
        down3 = self.down3(f)
        f = F.max_pool2d(down3, kernel_size=2, stride=2)
        down4 = self.down4(f)
        f = F.max_pool2d(down4, kernel_size=2, stride=2)
        down5 = self.down5(f)
        f = F.max_pool2d(down5, kernel_size=2, stride=2)

        f = self.same(f)

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        #f = F.max_unpool2d(f, i4, kernel_size=2, stride=2)
        f = self.up5(torch.cat([down5, f], 1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up4(torch.cat([down4, f], 1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up3(torch.cat([down3, f], 1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up2(torch.cat([down2, f], 1))

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up1(torch.cat([down1, f], 1))

        f = self.feature(f)
        f = F.dropout(f, p=0.5)
        logit = self.logit(f)

        return logit
