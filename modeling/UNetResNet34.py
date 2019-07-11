from torch import nn
from torch.nn import functional as F
import torch
import torchvision
from torchsummary import summary
from .BasicModule import *


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x

class SpatialGate(nn.Module):
    """docstring for SpatialGate"""
    def __init__(self, out_channels):
        super(SpatialGate, self).__init__()
        self.conv = nn.Conv2d(out_channels,1,kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        x = self.conv(x)
        return F.sigmoid(x)

class ChannelGate(nn.Module):
    """docstring for SpatialGate"""
    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(out_channels,out_channels//2,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(out_channels//2,out_channels,kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        x = nn.MaxPool2d(kernel_size=(x.size(2),x.size(3)))(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.sigmoid(self.conv2(x))
        return x
      
class ShortcutAttention(nn.Module):
    def __init__(self, out_channels):
        super(ShortcutAttention, self).__init__()
        self.spatial_gate = SpatialGate(out_channels)
        self.channel_gate = ChannelGate(out_channels)
    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1*x+g2*x
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.spatial_gate = SpatialGate(out_channels)
        self.channel_gate = ChannelGate(out_channels)
        
    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode='bilinear',
                       align_corners=True)  # False
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1*x+g2*x
        return x

#
# resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
# resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
# resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
# resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
# resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'


class UNetResNet34(BasicModule):
    def __init__(self, pretrained=False):
        super(UNetResNet34,self).__init__()
        if pretrained:
            print('loading pretrained model...')
        else:
            print('loading model without pretrained...')
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)

        self.num_classes = 4

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  # 128
        self.encoder4 = self.resnet.layer3  # 256
        self.encoder5 = self.resnet.layer4  # 512
        
        self.encoderAtten2 = ShortcutAttention(64)  # 64
        self.encoderAtten3 = ShortcutAttention(128)  # 128
        self.encoderAtten4 = ShortcutAttention(256)  # 256
        self.encoderAtten5 = ShortcutAttention(512)  # 512
        
        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder5 = Decoder(512+256, 512, 64)
        self.decoder4 = Decoder(256+64, 256, 64)
        self.decoder3 = Decoder(128+64, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)
        
        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  self.num_classes, kernel_size=1, padding=0),
        )
        

    def forward(self, x):
        #batch_size,C,H,W = x.shape

        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # x = torch.stack([
        #     (x-mean[0])/std[0],
        #     (x-mean[1])/std[1],
        #     (x-mean[2])/std[2],
        # ], 1)
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        e2 = self.encoder2(x)  # ; print('e2',e2.size())
        e3 = self.encoder3(e2)  # ; print('e3',e3.size())
        e4 = self.encoder4(e3)  # ; print('e4',e4.size())
        e5 = self.encoder5(e4)  # ; print('e5',e5.size())

        c = self.center(e5)

        d5 = self.decoder5(torch.cat([c, self.encoderAtten5(e5)], 1))  # ; print('d5',f.size())
        d4 = self.decoder4(torch.cat([d5, self.encoderAtten4(e4)], 1))  # ; print('d4',f.size())
        d3 = self.decoder3(torch.cat([d4, self.encoderAtten3(e3)], 1))  # ; print('d3',f.size())
        d2 = self.decoder2(torch.cat([d3, self.encoderAtten2(e2)], 1))  # ; print('d2',f.size())
        d1 = self.decoder1(d2)                      # ; print('d1',f.size())
        
        #hypercolumns
        f = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear',
                       align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear',
                       align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear',
                       align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False)), 1
        )
        f = F.dropout(f, p=0.5)

        logit = self.logit(f)
        return logit

if __name__ == '__main__':
    model = UNetResNet34()
    model.cuda()
    print(summary(model,(256,256)))