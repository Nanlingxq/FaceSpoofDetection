# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:14
# @Author : zhuying
# @Company : Minivision
# @File : MultiFTNet.py
# @Software : PyCharm
from torch import nn
import torch.nn.functional as F
import os
from src.model_lib.MiniFASNet import MiniFASNetV1,MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.model_lib.depth_auxiliary import DepthGenerator

def get_model_name(str):
    file_name = os.path.basename(str)
    name_without_ext = os.path.splitext(file_name)[0]
    model_name = name_without_ext.split('_')[-1]
    return model_name


class FTGenerator(nn.Module):
    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    def __init__(self, img_channel=3, num_classes=3, embedding_size=128, conv6_kernel=(5, 5), conf=None):
        super(MultiFTNet, self).__init__()
        self.conf = conf
        self.img_channel = img_channel
        self.num_classes = num_classes
        if self.conf.get('pretrain_model_src', None) is not None:
            model_name = get_model_name(self.conf.get('pretrain_model_src', None))
            if model_name == 'MiniFASNetV1':
                self.model = MiniFASNetV1(embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                                      num_classes=num_classes, img_channel=img_channel)
            elif model_name =="MiniFASNetV2":
                self.model = MiniFASNetV2(embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                                      num_classes=num_classes, img_channel=img_channel)
            elif model_name == 'MiniFASNetV1SE':
                self.model = MiniFASNetV1SE(embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                                      num_classes=num_classes, img_channel=img_channel)
            elif model_name == 'MiniFASNetV2SE':
                self.model = MiniFASNetV2SE(embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                                      num_classes=num_classes, img_channel=img_channel)
            else:
                raise ValueError(f"Unknown model name: {model_name}")
        else:
            self.model = MiniFASNetV2(embedding_size=embedding_size, conv6_kernel=conv6_kernel,
                                      num_classes=num_classes, img_channel=img_channel)
        self.FTGenerator = FTGenerator(in_channels=128)
        self.depth_aux_enabled = bool(self.conf.get('depth_aux_enabled', False))
        self.DepthGenerator = DepthGenerator(in_channels=128, output_size=self.conf.input_size) if self.depth_aux_enabled else None
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.drop(x1)
        cls = self.model.prob(x1)

        if self.training:
            ft = self.FTGenerator(x)
            if self.depth_aux_enabled:
                depth_map = self.DepthGenerator(x)
                return cls, ft, depth_map
            return cls, ft
        else:
            return cls
