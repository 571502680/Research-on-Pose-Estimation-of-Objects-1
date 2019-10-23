# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch.nn as nn

from models.layers.DUC import DUC
from models.layers.SE_Resnet import SEResnet

# Import training option
from opt import opt


def createModel():
    return FastPose_SE()


class FastPose_SE(nn.Module):
    conv_dim = 128

    def __init__(self):
        super(FastPose_SE, self).__init__()

        self.preact = SEResnet('resnet101')

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)

        self.conv_out = nn.Conv2d(
            self.conv_dim, opt.nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)#[1,2048,10,8]
        out = self.suffle1(out)#[1,512,20,16]
        out = self.duc1(out)#[1,256,40,32]
        out = self.duc2(out)#[1,256,80,64]

        out = self.conv_out(out)#[1,17,80,64]
        return out
