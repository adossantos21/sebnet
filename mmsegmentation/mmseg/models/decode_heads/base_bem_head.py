# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from mmseg.models.utils import BEM
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

@MODELS.register_module()
class BaselineBEMHead(BaseDecodeHead):
    """Baseline + BEM head for mapping feature to a predefined set
    of classes.

    Args:
        in_channels (int): Number of feature maps coming from 
        the decoded prediction.
            Default: 256.
        num_classes (int): Number of classes in the training
        dataset.
            Default: 19 for Cityscapes.
    """

    def __init__(self, in_channels=256, num_classes=19, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(in_channels, int)
        assert isinstance(num_classes, int)
        self.in_channels = in_channels
        self.num_classes = num_classes
        if self.training:
            self.bem = BEM(planes=self.in_channels // 4)
            self.side5_head = nn.Conv2d(self.in_channels // 2, self.num_classes, kernel_size=1)
            self.fuse_head = nn.Conv2d(self.in_channels // 2, self.num_classes, kernel_size=1)
        self.seg_head = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward function.
        x should be a tuple of outputs:
        x_1, x_2, x_3, x_4, x_5, x_7 = x
        x_1 has shape (N, 64, H/4, W/4)
        x_2 has shape (N, 128, H/8, W/8)
        x_3 has shape (N, 256, H/16, W/16)
        x_4 has shape (N, 512, H/32, W/32)
        x_5 has shape (N, 1024, H/64, W/64)
        x_7 has shape (N, 256, H/8, W/8) and should have been processed through the DAPPM Neck
        """
        if self.training:
            side5, fuse = self.bem(x) # side5 and fuse (N, C=128, H/8, W/8)
            side5 = self.side5_head(side5)
            fuse = self.fuse_head(fuse)
            output = self.seg_head(x[5])
            return tuple([output, side5, fuse])
        else:
            output = self.seg_head(x[5])
            return output