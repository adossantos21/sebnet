# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.models.utils import BaseSegHead
from mmseg.models.utils import DFF_EarlierLayers as DFF
import torch
import torch.nn.functional as F
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

from typing import Tuple
from mmseg.utils import OptConfigType, SampleList
from torch import Tensor

@MODELS.register_module()
class BaselineDFFHeadEarlierLayers(BaseDecodeHead):
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

    def __init__(self, 
                 in_channels=256, 
                 num_classes=19, 
                 norm_cfg: OptConfigType = dict(type='SyncBN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 eval_edges: bool = False,
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
        self.eval_edges = eval_edges
        self.dff = DFF(nclass=self.num_classes)
        self.seg_head = BaseSegHead(in_channels, in_channels, self.stride, norm_cfg, act_cfg)

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
            side5, fuse = self.dff(x) # side5: (N, K, H/4, W/4), fuse: (N, K, H/4, W/4), where K is the number of classes in the labeled dataset
            x[-1] = F.interpolate(
                x[-1],
                size=x[1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = self.seg_head(x[-1], self.cls_seg) # (N, K, H/8, W/8)
            return tuple([output, side5, fuse])
        else:
            if self.eval_edges:
                _, output = self.dff(x)
                output = tuple([output])
            else:
                x[-1] = F.interpolate(
                    x[-1],
                    size=x[1].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                output = self.seg_head(x[-1], self.cls_seg)
            return output

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = [
            data_sample.gt_multi_edge_map.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        return gt_sem_segs, gt_edge_segs

    def loss_by_feat(self, logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        seg_logits, side5_logits, fuse_logits = logits
        seg_label, bd_multi_label = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
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
        seg_label = seg_label.squeeze(1)
        bd_multi_label = bd_multi_label.squeeze(1)
        logits = dict(
            seg_logits=seg_logits,
            side5_logits=side5_logits,
            fuse_logits=fuse_logits
        )
        loss = dict()
        loss['loss_seg'] = self.loss_decode[0](seg_logits, seg_label)
        loss['loss_side5'] = self.loss_decode[1](side5_logits, bd_multi_label)
        loss['loss_fuse'] = self.loss_decode[2](fuse_logits, bd_multi_label)
        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss, logits