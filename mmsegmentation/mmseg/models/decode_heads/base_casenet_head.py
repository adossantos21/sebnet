# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.models.utils import CASENet
import torch
import torch.nn as nn
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

from typing import List, Tuple, Optional
from mmseg.utils import OptConfigType, SampleList
from torch import Tensor

from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule

class BasePIDHead(BaseModule):
    """Base class for PID head.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict or list[dict], optional): Init config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            order=('norm', 'act', 'conv'))
        _, self.norm = build_norm_layer(norm_cfg, num_features=channels)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x: Tensor, cls_seg: Optional[nn.Module]) -> Tensor:
        """Forward function.
        Args:
            x (Tensor): Input tensor.
            cls_seg (nn.Module, optional): The classification head.

        Returns:
            Tensor: Output tensor.
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if cls_seg is not None:
            x = cls_seg(x)
        return x

@MODELS.register_module()
class BaselineCASENetHead(BaseDecodeHead):
    """Baseline + CASENet head for mapping feature to a predefined set
    of classes.

    Args:
        in_channels (int): Number of feature maps coming from 
        the decoded prediction.
            Default: 256.
        num_classes (int): Number of classes in the training
        dataset.
            Default: 19 for Cityscapes.
    """

    def __init__(self, 
                 in_channels=256, 
                 num_classes=19, 
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            in_channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        assert isinstance(in_channels, int)
        assert isinstance(num_classes, int)
        self.in_channels = in_channels
        self.num_classes = num_classes
        if self.training:
            self.casenet = CASENet(nclass=self.num_classes)
        self.seg_head = BasePIDHead(in_channels, in_channels, norm_cfg, act_cfg)

    def forward(self, x):
        """
        Forward function.
        x should be a tuple of outputs:
        x_0, x_1, x_2, x_3, x_4, x_out = x
        x_0 has shape (N, 64, H/4, W/4)
        x_1 has shape (N, 128, H/8, W/8)
        x_2 has shape (N, 256, H/16, W/16)
        x_3 has shape (N, 512, H/32, W/32)
        x_4 has shape (N, 1024, H/64, W/64)
        x_out has shape (N, 256, H/8, W/8)
        """
        if self.training:
            side5, fuse = self.casenet(x) # side5: (N, C=Num_Classes, H/8, W/8), fuse: (N, C=Num_Classes, H/8, W/8)
            output = self.seg_head(x[-1], self.cls_seg)
            return [output, side5, fuse]
        else:
            output = self.seg_head(x[-1], self.cls_seg)
            return output
        
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = [
            data_sample.gt_edge_map.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        return gt_sem_segs, gt_edge_segs

    def loss_by_feat(self, seg_logits: List[Tensor],
                     batch_data_samples: SampleList) -> dict:
        output_logits, side5_logits, fuse_logits = seg_logits
        sem_label, bd_multi_label = self._stack_batch_gt(batch_data_samples)
        output_logits = resize(
            input=output_logits,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        side5_logits = resize(
            input=side5_logits,
            size=bd_multi_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)
        fuse_logits = resize(
            input=fuse_logits,
            size=bd_multi_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)
        sem_label = sem_label.squeeze(1)
        bd_multi_label = bd_multi_label.squeeze(1)
        #print(f"torch.unique(bd_multi_label): {torch.unique(bd_multi_label)}")
        #tmp = bd_multi_label[0]
        #for tmp in bd_multi_label:
        #    for cls in tmp:
        #        print(f"torch.unique(cls): {torch.unique(cls)}")
        #import sys
        #sys.exit()
        logits = dict(
            seg_logits=output_logits,
            side5_logits=side5_logits,
            fuse_logits=fuse_logits
        )
        loss = dict()
        loss['loss_ce'] = self.loss_decode[0](output_logits, sem_label)
        loss['loss_side5'] = self.loss_decode[1](side5_logits, bd_multi_label)
        loss['loss_fuse'] = self.loss_decode[2](fuse_logits, bd_multi_label)
        loss['acc_seg'] = accuracy(
            output_logits, sem_label, ignore_index=self.ignore_index)
        return loss, logits