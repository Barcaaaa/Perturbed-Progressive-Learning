import torch
import torch.nn as nn
import torch.nn.functional as F

from model.semseg.base import BaseNet

class FCN(BaseNet):
    def __init__(self, backbone, numClass):
        super(FCN, self).__init__(backbone)
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self._up_kwargs = {'mode': 'bilinear'}
        self.max_patch_side = 960
        # self.gcsnn = gcsnn.GSCNN(numClass)
        # Res net
        # self.resnet_g = resnet18(True)
        # self.resnet_g2 = resnet50(True)

        # fpn module
        if backbone == 'resnet18':
            self.fpn_g = fpn_module_local(numClass)
        elif backbone == 'resnet50':
            self.fpn_g = fpn_module_global(numClass)
        # self.fpn_g2 = fpn_module_global(numClass)
        self.outlayer = nn.Linear(512, 5)
        self.inference_time = 0
        self.count = 0

    def get_inference_time(self):
        return self.inference_time

    def base_forward(self, input):
        # torch.cuda.synchronize()
        # start = time.time()

        _, _, h, w = input.shape
        # c2_l, c3_l, c4_l, c5_l = self.resnet_g.forward(input)
        c2_l, c3_l, c4_l, c5_l = self.backbone.base_forward(input)
        output_l, output_fea_l = self.fpn_g.forward(c2_l, c3_l, c4_l, c5_l)

        # end = time.time()
        # self.count = self.count + 1
        # self.inference_time += (end - start)
        # print(self.inference_time,self.count)
        # x = F.avg_pool2d(c5_l, 16)
        # x = x.view(x.size(0), -1)
        # out = self.outlayer(x)
        out = F.interpolate(output_l, size=(h,w), mode="bilinear", align_corners=True)

        return out


class fpn_module_local(nn.Module):  # resnet18
    def __init__(self, numClass):
        super(fpn_module_local, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self._up_kwargs = {'mode': 'bilinear'}
        # global branch
        # Top layer
        self.toplayer = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0) # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.g2l_c5 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        # self.ps3_up = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.up_regular = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        # Classify layers
        self.classify = nn.Conv2d(32*4, numClass, kernel_size=3, stride=1, padding=1)

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)

        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        out = torch.cat([p5, p4, p3, p2], dim=1)
        return out

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def forward(self, c2, c3, c4, c5, c5_g2l=None):
        # global
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p5 = self.smooth1_1(p5)
        p4 = self.smooth2_1(p4)
        p3 = self.smooth3_1(p3)
        p2 = self.smooth4_1(p2)

        p5 = self.smooth1_2(p5)
        p4 = self.smooth2_2(p4)
        p3 = self.smooth3_2(p3)
        p2 = self.smooth4_2(p2)
        # Classify
        ps3 = self._concatenate(p5, p4, p3, p2)
        output = self.classify(ps3)
        return output, ps3


class fpn_module_global(nn.Module):
    def __init__(self, numClass):
        super(fpn_module_global, self).__init__()
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self._up_kwargs = {'mode': 'bilinear'}
        # global branch
        # Top layer
        # self.conv_3x3 = nn.Sequential(
        #     nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # )

        # self.ocr_gather_head = SpatialGather_Module(numClass)
        #
        # self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
        #                                          key_channels=256,
        #                                          out_channels=512,
        #                                          scale=1,
        #                                          dropout=0.05,
        #                                          )
        # self.conv_bn_dropout = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1)
        # )

        # out_channels=256)


        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # Classify layers

        self.classify = nn.Conv2d(128*4, numClass, kernel_size=3, stride=1, padding=1)
        self.numclass = numClass
        self.l2g_c5 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)

        # self.dsn_head = nn.Sequential(
        #     nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(0.1),
        #     nn.Conv2d(512, self.numclass, kernel_size=1, stride=1, padding=0, bias=True)
        # )
        self.aux_head = nn.Sequential(
            nn.Conv2d(512, 512,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=512),
            nn.Conv2d(512, self.numclass,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def _concatenate(self, p5, p4, p3, p2):
        _, _, H, W = p2.size()
        p3 = F.interpolate(p3, size=(H, W), **self._up_kwargs)

        p5 = F.interpolate(p5, size=(H, W), **self._up_kwargs)
        p4 = F.interpolate(p4, size=(H, W), **self._up_kwargs)
        out = torch.cat([p5, p4, p3, p2], dim=1)
        return out

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), **self._up_kwargs) + y

    def forward(self, c2, c3, c4, c5, c5_l2g=None):
        # global
        # Top-down
        # c5 = self.conv_3x3(c5)
        # ocp_feats = self.ocp_gather_infer(c5, self.numclass,global_res)
        #
        # p5 = self.ocp_distr_infer(c5, ocp_feats,global_res)
        # p5 = self.conv_bn_dropout(p5)

        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p5 = self.smooth1_1(p5)
        p4 = self.smooth2_1(p4)
        p3 = self.smooth3_1(p3)
        p2 = self.smooth4_1(p2)

        p5 = self.smooth1_2(p5)
        p4 = self.smooth2_2(p4)
        p3 = self.smooth3_2(p3)
        p2 = self.smooth4_2(p2)
        # Classify
        ps3 = self._concatenate(p5, p4, p3, p2)


        # out_aux = self.aux_head(ps3)
        # # compute contrast feature
        # # ps3 = self.conv3x3_ocr(ps3)
        #
        # context = self.ocr_gather_head(ps3, out_aux)
        # ps3 = self.ocr_distri_head(ps3, context)

        output = self.classify(ps3)
        return output,ps3
