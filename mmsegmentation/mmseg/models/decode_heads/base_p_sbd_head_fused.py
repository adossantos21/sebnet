# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.models.utils import BaseSegHead, Bag, PModule
'''
from mmseg.models.utils import(
    CASENet_Original as CASENet,
    DFF_Original as DFF,
    BEM_Original as BEM
)
'''
from mmseg.models.utils import(
    CASENet,
    DFF,
    BEM
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

from typing import Tuple
from mmseg.utils import OptConfigType, SampleList
from torch import Tensor

@MODELS.register_module()
class BagBaselinePSBDHead(BaseDecodeHead):
    """Baseline + P Branch + SBD head for mapping feature to a predefined set
    of classes (with bag fusion).

    Args:
        in_channels (int): Number of feature maps coming from 
        the decoded prediction.
            Default: 256.
        num_classes (int): Number of classes in the training
        dataset.
            Default: 19 for Cityscapes.
    """

    def __init__(self, 
                 in_channels: int = 256, 
                 num_classes: int = 19, 
                 num_stem_blocks: int = 3,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 sbd_head='casenet', 
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
        self.stride = 1
        self.num_stem_blocks = num_stem_blocks
        self.sbd_head = sbd_head
        self.p_module = PModule(channels=self.in_channels // 4, num_stem_blocks=self.num_stem_blocks)
        self.p_head = BaseSegHead(self.in_channels // 2, self.in_channels, self.stride, norm_cfg, act_cfg)
        self.p_cls_seg = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)
        self.side5_cls_seg = nn.Conv2d(in_channels // 2, self.num_classes, kernel_size=1)
        self.fuse_cls_seg = nn.Conv2d(in_channels, self.num_classes, kernel_size=1)
        self.fusion = Bag(self.in_channels, self.in_channels, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.seg_head = BaseSegHead(self.in_channels, self.in_channels, self.stride, norm_cfg, act_cfg)
        if self.sbd_head == 'casenet' or self.sbd_head == 'dff':
            self.sbd = CASENet(nclass=self.num_classes) if self.sbd_head=='casenet' else DFF(nclass=self.num_classes)
            self.fuse_head = BaseSegHead(self.num_classes, in_channels, self.stride, norm_cfg, act_cfg)
            self.side5_head = BaseSegHead(self.num_classes, in_channels // 2, self.stride, norm_cfg, act_cfg)
        elif self.sbd_head == 'bem':
            self.sbd = BEM(planes=self.in_channels // 4)
            self.fuse_head = BaseSegHead(in_channels // 2, in_channels, self.stride, norm_cfg, act_cfg)
            self.side5_head = BaseSegHead(in_channels // 2, in_channels // 2, self.stride, norm_cfg, act_cfg)
        else:
            raise ValueError(f"Invalid SBD Head. self.sbd_head should be one of ['casenet', 'dff', 'bem']; instead it is {self.sbd_head}")
        

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
        x_out has shape (N, 256, H/64, W/64)
        """
        if self.training:
            temp_p, x_p = self.p_module(x)
            p_supervised = self.p_head(temp_p, self.p_cls_seg)
            side5, fuse = self.sbd(x)
            if self.sbd_head == 'bem':
                side5 = self.side5_head(side5, self.side5_cls_seg)
                fuse_fusion = self.fuse_head(fuse, None)
                fuse = self.fuse_cls_seg(fuse_fusion)
            if self.sbd_head == 'casenet' or self.sbd_head == 'dff':
                side5 = self.side5_head(side5, self.side5_cls_seg)
                fuse_fusion = self.fuse_head(fuse, None)
                fuse = self.fuse_cls_seg(fuse_fusion)
        else:
            x_p = self.p_module(x)
            _, fuse = self.sbd(x)
            fuse_fusion = self.fuse_head(fuse, None)
        x[-1] = F.interpolate(
            x[-1],
            size=x[1].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        #print(f"x_p shape: {x_p.shape}")
        #print(f"x[-1] shape: {x[-1].shape}")
        #print(f"fuse_fusion shape: {fuse_fusion.shape}")
        #import sys
        #sys.exit()
        feats = self.fusion(x_p, x[-1], fuse_fusion)
        output = self.seg_head(feats, self.cls_seg)
        if self.training:
            return tuple([output, p_supervised, side5, fuse])
        else:
            return output
        
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_multi_edge_segs = [
            data_sample.gt_multi_edge_map.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_multi_edge_segs = torch.stack(gt_multi_edge_segs, dim=0)
        return gt_sem_segs, gt_multi_edge_segs

    def loss_by_feat(self, logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        seg_logits, p_logits, side5_logits, fuse_logits = logits
        sem_label, bd_multi_label = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(
            input=seg_logits,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        p_logits = resize(
            input=p_logits,
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
        logits = dict(
            seg_logits=seg_logits,
            p_logits=p_logits,
            side5_logits=side5_logits,
            fuse_logits=fuse_logits
        )
        loss['loss_sem'] = self.loss_decode[0](seg_logits, sem_label)
        loss['loss_p'] = self.loss_decode[1](p_logits, sem_label)
        loss['loss_side5'] = self.loss_decode[2](side5_logits, bd_multi_label)
        loss['loss_fuse'] = self.loss_decode[3](fuse_logits, bd_multi_label)
        loss['acc_seg'] = accuracy(
            seg_logits, sem_label, ignore_index=self.ignore_index)
        return loss, logits