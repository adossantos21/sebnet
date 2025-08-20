# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.models.utils import DFF
import torch.nn as nn

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

@MODELS.register_module()
class BaselineDFFHead(BaseDecodeHead):
    """Baseline + DFF head for mapping feature to a predefined set
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
            self.dff = DFF(nclass=self.num_classes)
        self.seg_head = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward function.
        x should be a tuple of outputs:
        x_1, x_2, x_3, x_4, x_5, x_out = x
        x_1 has shape (N, 64, H/4, W/4)
        x_2 has shape (N, 128, H/8, W/8)
        x_3 has shape (N, 256, H/16, W/16)
        x_4 has shape (N, 512, H/32, W/32)
        x_5 has shape (N, 1024, H/64, W/64)
        x_out has shape (N, 256, H/8, W/8)
        """
        if self.training:
            side5, fuse = self.dff(x) # side5: (N, C=Num_Classes, H/8, W/8), fuse: (N, C=Num_Classes, H/8, W/8)
            output = self.seg_head(x[5])
            return tuple([output, side5, fuse])
        else:
            output = self.seg_head(x[5])
            return output