# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from .base_backbone import BaseBackbone
from mmpretrain.registry import MODELS
from mmengine.runner import CheckpointLoader
#from mmpretrain.utils import OptConfigType
from mmpretrain.models.utils import BasicBlock, Bottleneck
from mmpretrain.models.utils.basic_block import OptConfigType

@MODELS.register_module()
class SEBNet(BaseBackbone):
    """SEBNet backbone.

    This backbone is the implementation of `SEBNet: Real-Time Semantic
    Segmentation with Semantic Boundary Detection Conditioning.

    Licensed under the MIT License.

    Args:
        in_channels (int): The number of input channels. Default: 3.
        channels (int): The number of channels in the stem layer. Default: 64.
        ppm_channels (int): The number of channels in the PPM layer.
            Default: 96.
        num_stem_blocks (int): The number of blocks in the stem layer.
            Default: 2.
        num_branch_blocks (int): The number of blocks in the branch layer.
            Default: 3.
        align_corners (bool): The align_corners argument of F.interpolate.
            Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 64,
                 ppm_channels: int = 96,
                 num_stem_blocks: int = 2,
                 num_branch_blocks: int = 3,
                 align_corners: bool = False,
                 norm_cfg: dict = dict(type='BN'),
                 act_cfg: dict = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super(SEBNet, self).__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stem layer - we need better granularity to integrate the SBD modules
        self.conv1 =  nn.Sequential(
             ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )
        self.stage_1 = self._make_layer(
            block=BasicBlock,
            in_channels=channels,
            channels=channels,
            num_blocks=num_stem_blocks)
        self.stage_2 = self._make_layer(
            block=BasicBlock,
            in_channels=channels,
            channels=channels * 2,
            num_blocks=num_stem_blocks,
            stride=2)
        self.relu = nn.ReLU()

        # I Branch
        self.i_branch_layers = nn.ModuleList()
        for i in range(3):
            self.i_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2**(i + 1),
                    channels=channels * 8 if i > 0 else channels * 4,
                    num_blocks=num_branch_blocks if i < 2 else 2,
                    stride=2))
        
    def _make_stem_layer(self, in_channels: int, channels: int,
                         num_blocks: int) -> nn.Sequential:
        """Make stem layer.

        Args:
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.

        Returns:
            nn.Sequential: The stem layer.
        """

        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.append(
            self._make_layer(BasicBlock, channels, channels, num_blocks))
        layers.append(nn.ReLU())
        layers.append(
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _make_layer(self,
                    block: BasicBlock,
                    in_channels: int,
                    channels: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Make layer for PIDNet backbone.
        Args:
            block (BasicBlock): Basic block.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Sequential: The Branch Layer.
        """
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)

        layers = [block(in_channels, channels, stride, downsample)]
        in_channels = channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels,
                    channels,
                    stride=1,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))
        return nn.Sequential(*layers)

    def _make_single_layer(self,
                           block: Union[BasicBlock, Bottleneck],
                           in_channels: int,
                           channels: int,
                           stride: int = 1) -> nn.Module:
        """Make single layer for PIDNet backbone.
        Args:
            block (BasicBlock or Bottleneck): Basic block or Bottleneck.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Module
        """

        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
        return block(
            in_channels, channels, stride, downsample, act_cfg_out=None)

    def init_weights(self):
        """Initialize the weights in backbone.

        Since the D branch is not initialized by the pre-trained model, we
        initialize it with the same method as the ResNet.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(ckpt, strict=False)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
        w_out = x.shape[-1] // 8
        h_out = x.shape[-2] // 8

        # stage 0
        x = self.conv1(x) # (N, C=64, H/4, W/4)

        # stage 1
        x_1 = self.relu(self.stage_1(x)) # (N, C=64, H/4, W/4)

        # stage 2
        x_2 = self.relu(self.stage_2(x_1)) # (N, C=128, H/8, W/8)

        # stage 3
        x_3 = self.relu(self.i_branch_layers[0](x_2)) # (N, C=256, H/16, W/16)

        # stage 4
        x_4 = self.relu(self.i_branch_layers[1](x_3)) # (N, C=512, H/32, W/32)

        # stage 5
        x_5 = self.i_branch_layers[2](x_4) # (N, C=1024, H/64, W/64)

        return x_5
        '''
        x_spp = self.spp(x_5) # performs adaptive avg pooling at several scaled kernels: {5, 9, 17}
        x_out = F.interpolate( # (N, 256, H/8, W/8)
            x_spp,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        
        if self.training: # self.training is inherent to MMSeg configurations throughout BaseModule and BaseDecodeHead objects.
            if config.ABLATION == 0:
                return x_out
            elif config.ABLATION == 1 or config.ABLATION == 2 or config.ABLATION == 4: # PI Model (1), ID Model (2), PID Model (4)
                return (x_2, x_3, x_4, x_out)
            else: # I SBD Model (3), PI SBD Model (5), ID SBD Model (6), PID SBD Model (7)
                return (x_1, x_2, x_3, x_4, x_5, x_out)
        else:
            return x_out
        '''


