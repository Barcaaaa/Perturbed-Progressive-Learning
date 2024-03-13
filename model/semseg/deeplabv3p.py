from model.semseg.base import BaseNet

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus_plus(BaseNet):
    # 添加了8个膨胀卷积
    def __init__(self, backbone, nclass):
        super(DeepLabV3Plus_plus, self).__init__(backbone)

        low_level_channels = self.backbone.channels[0]
        high_level_channels = self.backbone.channels[-1]

        self.head = ASPPModule_plus(high_level_channels, range(9))

        # 1x1 conv
        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        # 3x3 conv
        self.fuse = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))

        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        c1, _, _, c4 = self.backbone.base_forward(x)

        c4 = self.head(c4) #ASPP
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)  # low level feature

        out = torch.cat([c1, c4], dim=1)
        out = self.fuse(out) #2,256,128,128

        out = self.classifier(out) #2 4 128 128

        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)#2 4 512 512

        return out

class DeepLabV3Plus(BaseNet):
    def __init__(self, backbone, nclass):
        super(DeepLabV3Plus, self).__init__(backbone)

        low_level_channels = self.backbone.channels[0]
        high_level_channels = self.backbone.channels[-1]

        self.head = ASPPModule(high_level_channels, (12, 24, 36))

        # 1x1 conv
        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        # 3x3 conv
        self.fuse = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))

        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        c1, _, _, c4 = self.backbone.base_forward(x)

        c4 = self.head(c4) #ASPP
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)  # low level feature

        out = torch.cat([c1, c4], dim=1)
        out = self.fuse(out) #2,256,128,128

        out = self.classifier(out) #2 4 128 128

        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)#2 4 5112 512

        return out


class DeepLabV3Plus_plus_decoder(BaseNet):
    # 在plus基础上修改了decoder
    def __init__(self, backbone, nclass):
        super(DeepLabV3Plus_plus_decoder, self).__init__(backbone)

        low_level_channels = self.backbone.channels[0]
        high_level_channels = self.backbone.channels[-1]

        self.head = ASPPModule_plus(high_level_channels, range(9))

        # 1x1 conv
        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        # 3x3 conv
        self.fuse1 = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))

        self.fuse2 = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))
        self.classifier1 = nn.Conv2d(256, nclass, 1, bias=True)
        self.classifier2 = nn.Conv2d(256, nclass, 1, bias=True)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        c1, _, _, c4 = self.backbone.base_forward(x)

        c4 = self.head(c4) #ASPP
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)  # low level feature

        out = torch.cat([c1, c4], dim=1)
        out1 = self.fuse1(out) #2,256,128,128
        out1 = self.classifier1(out1) #2 4 128 128

        out2 = self.fuse1(out) #2,256,128,128
        out2 = self.classifier2(out2) #2 4 128 128

        out = F.interpolate((out1+out2)/2, size=(512,512), mode="bilinear", align_corners=True)#2 4 5112 512

        return out

def ASPPConv(in_channels, out_channels, atrous_rate):
    # deeplabv3: BN add to ASPP
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block

def ASPPConv_plus(in_channels, out_channels, atrous_rate):
    # deeplabv3: BN add to ASPP
    # ASPP后在接一个1x1conv
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True),
                          nn.Conv2d(out_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True)
                          )

    return block

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        # 全局池化层
        self.b4 = ASPPPooling(in_channels, out_channels)
        # 1x1 conv
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)

class ASPPModule_plus(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule_plus, self).__init__()
        out_channels = in_channels // 8

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv_plus(in_channels, out_channels, atrous_rates[1])
        self.b2 = ASPPConv_plus(in_channels, out_channels, atrous_rates[2])
        self.b3 = ASPPConv_plus(in_channels, out_channels, atrous_rates[3])
        self.b4 = ASPPConv_plus(in_channels, out_channels, atrous_rates[4])
        self.b5 = ASPPConv_plus(in_channels, out_channels, atrous_rates[5])
        self.b6 = ASPPConv_plus(in_channels, out_channels, atrous_rates[6])
        self.b7 = ASPPConv_plus(in_channels, out_channels, atrous_rates[7])

        # 全局池化层
        self.b8 = ASPPPooling(in_channels, out_channels)

        # 1x1 conv
        self.project = nn.Sequential(nn.Conv2d(9 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        feat5 = self.b5(x)
        feat6 = self.b6(x)
        feat7 = self.b7(x)
        feat8 = self.b8(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8), 1)
        return self.project(y)

if __name__ == '__main__':
    model = DeepLabV3Plus("resnet18", 4)
    model.eval()
    x = torch.rand((2, 3, 512, 512))
    y = model(x)
    print(y.shape)
