import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmseg.models.losses import accuracy
from mmseg.models.utils import (
    resize,
    BaseSegHead,
    PModuleFused as PModule,
    EdgeModuleConditioned as EdgeModule,
    PIFusion as PreHead
)
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList
from .decode_head import BaseDecodeHead
from typing import Tuple

@MODELS.register_module()
class Ablation39(BaseDecodeHead):
    """
    Ablation 39 - Baseline + P Head (Fused) + Edge Head (Conditioned),
    with HED, SBD, and BAS supervisory signals. Fusion present for 
    P Head. See Holistically-Nested Edge Detection at 
    https://arxiv.org/pdf/1504.06375, Semantic Boundary Detection at 
    https://arxiv.org/pdf/1705.09759, and Boundary-Awareness at 
    https://arxiv.org/pdf/2206.02066 for more details.

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
                 stride: int = 1,
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
        assert isinstance(in_channels, int), f"Expected in_channels to be int, got {type(in_channels)}"
        assert isinstance(num_classes, int), f"Expected num_classes to be int, got {type(num_classes)}"
        assert isinstance(num_stem_blocks, int), f"Expected num_stem_blocks to be int, got {type(num_stem_blocks)}"
        assert isinstance(stride, int), f"Expected stride to be int, got {type(stride)}"
        self.eval_edges = eval_edges
        if self.training:
            self.p_head = BaseSegHead(in_channels // 2, in_channels, stride=stride, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.p_cls_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        if self.training or self.eval_edges:
            self.edge_module = EdgeModule(channels=in_channels // 4, num_stem_blocks=num_stem_blocks, eval_edges=self.eval_edges)
            self.hed_head = BaseSegHead(in_channels // 2, in_channels // 4, stride=stride, norm_cfg=norm_cfg) # No act_cfg here on purpose. See pidnet head.
            self.hed_cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
            self.sbd_head = BaseSegHead(in_channels // 2, in_channels // 4, stride=stride, norm_cfg=norm_cfg) # No act_cfg here on purpose. See pidnet head.
            self.sbd_cls_seg = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)
        self.p_module = PModule(channels=in_channels // 4, num_stem_blocks=num_stem_blocks)
        self.pre_head = PreHead(in_channels, in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.seg_head = BaseSegHead(in_channels, in_channels, stride=stride, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
    def forward(self, x: Tuple[Tensor, ...]):
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
            x_p_feats, x_p = self.p_module(x) # x_p_feats: (N, 128, H/8, W/8), x_p: (N, 256, H/8, W/8)
            x_edges = self.edge_module(x) # x_edges: (N, 128, H/8, W/8)
            x[-1] = F.interpolate(
                x[-1],
                size=x[1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            x_p_supervised = self.p_head(x_p_feats, self.p_cls_seg) # (N, K, H/8, W/8), where K is the number of classes in the labeled dataset
            hed = self.hed_head(x_edges, self.hed_cls_seg) # (N, 1, H/8, W/8)
            sbd = self.sbd_head(x_edges, self.sbd_cls_seg) # (N, K, H/8, W/8)
            feats = self.pre_head(x_p, x[-1])
            output = self.seg_head(feats, self.cls_seg) # (N, K, H/8, W/8)
            return tuple([output, x_p_supervised, hed, sbd])
        else:
            if self.eval_edges:
                x_edges = self.edge_module(x)
                hed = self.hed_head(x_edges, self.hed_cls_seg)
                sbd = self.sbd_head(x_edges, self.sbd_cls_seg)
                output = tuple([hed, sbd])
            else:
                x_p = self.p_module(x)
                x[-1] = F.interpolate(
                    x[-1],
                    size=x[1].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                feats = self.pre_head(x_p, x[-1])
                output = self.seg_head(feats, self.cls_seg)
            return output

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = [
            data_sample.gt_edge_map.data for data_sample in batch_data_samples
        ]
        gt_multi_edge_segs = [
            data_sample.gt_multi_edge_map.data for data_sample in batch_data_samples
        ]
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        gt_multi_edge_segs = torch.stack(gt_multi_edge_segs, dim=0)
        return gt_sem_segs, gt_edge_segs, gt_multi_edge_segs

    def loss_by_feat(self, logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        seg_logits, p_logits, hed_logits, sbd_logits = logits
        seg_label, hed_label, sbd_label = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        p_logits = resize(
            input=p_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        hed_logits = resize(
            input=hed_logits,
            size=hed_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        sbd_logits = resize(
            input=sbd_logits,
            size=sbd_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        hed_label = hed_label.squeeze(1)
        sbd_label = sbd_label.squeeze(1)
        logits = dict(
            seg_logits=seg_logits,
            p_logits=p_logits,
            hed_logits=hed_logits,
            sbd_logits=sbd_logits
        )
        loss = dict()
        loss['loss_seg'] = self.loss_decode[0](seg_logits, seg_label)
        loss['loss_seg_p'] = self.loss_decode[1](p_logits, seg_label)
        loss['loss_hed'] = self.loss_decode[2](hed_logits, hed_label)
        loss['loss_sbd'] = self.loss_decode[3](sbd_logits, sbd_label)
        filler = torch.ones_like(seg_label) * self.ignore_index
        seg_hed_label = torch.where(
            torch.sigmoid(hed_logits[:, 0, :, :]) > 0.8, seg_label, filler)
        loss['loss_bas'] = self.loss_decode[4](seg_logits, seg_hed_label)
        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss, logits