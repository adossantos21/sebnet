'''
Auxiliary modules for semantic segmentation models.
For PIDNet, we have the P Branch and the D Branch modules.
For Semantic Boundary Detection (SBD), we have the CASENet, DFF, and BGF modules.
'''

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from torch import Tensor

from mmseg.utils import OptConfigType
from mmseg.models.utils import BasicBlock
from mmseg.models.utils import BottleneckExp2 as Bottleneck
from .base import CustomBaseModule
from .convnext_block import ConvNeXtBlock
from .fusion_modules import PagFM

from mmengine.model import BaseModule

class BaseSegHead(BaseModule):
    """Base class for segmentation heads.

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
                 stride: int = 1,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.conv = ConvModule(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
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

class PModuleScaled(CustomBaseModule):
    """
    Model layers for the P branch of PIDNet.
    """
    arch_settings = {
        'small': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [128, 128, 256],  # Output channels per stage [stage3, stage4, stage5]
            'depths': [2, 2, 1],
        },
        'small_scaled': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [128, 256, 512],
            'depths': [2, 2, 1],
        },
        'base': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [128, 128, 256],
            'depths': [3, 3, 1],
        },
        'base_scaled': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [128, 256, 512],
            'depths': [3, 3, 1],
        },
        'large': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [128, 128, 256],
            'depths': [3, 3, 1],
        },
        'large_scaled': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [128, 256, 512],
            'depths': [3, 3, 1],
        },
        'xlarge': {
            'backbone_channels': [96, 192, 384, 768, 1536],
            'branch_channels': [192, 192, 384],
            'depths': [3, 3, 1],
        },
        'xlarge_scaled': {
            'backbone_channels': [96, 192, 384, 768, 1536],
            'branch_channels': [192, 384, 768],
            'depths': [3, 3, 1],
        },
    }

    def __init__(self,
                 arch: Union[str, dict] = 'small',
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='SyncBN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Arch "{arch}" not found. Choose from {set(self.arch_settings.keys())} ' \
                f'or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            required = ['backbone_channels', 'branch_channels', 'depths']
            assert all(k in arch for k in required), \
                f'Custom arch dict must have {required}.'
            assert len(arch['branch_channels']) == 3, \
                f'branch_channels must have 3 elements, got {len(arch["branch_channels"])}.'
        
        self.backbone_channels = arch['backbone_channels']
        self.branch_channels = arch['branch_channels']  # [stage3_out, stage4_out, stage5_out]
        self.depths = arch['depths']
        
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.relu = nn.ReLU()

        # P Branch layers
        self.p_branch_layers = nn.ModuleList()
        for i in range(3):
            block = BasicBlock if i < 2 else Bottleneck
            
            # Determine input channels
            if i == 0:
                in_ch = self.backbone_channels[1]  # x_1 channels
            else:
                in_ch = self.branch_channels[i - 1]  # Previous stage output
            
            # Determine layer channels (accounting for block expansion)
            out_ch = self.branch_channels[i]
            layer_ch = out_ch // block.expansion
            
            self.p_branch_layers.append(
                self._make_layer(
                    block=block,
                    in_channels=in_ch,
                    channels=layer_ch,
                    num_blocks=self.depths[i]))
        
        # Compression layers (backbone -> branch channels)
        self.compression_1 = ConvModule(
            self.backbone_channels[2],    # x_2 channels
            self.branch_channels[0],      # Match stage 3 output
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_2 = ConvModule(
            self.backbone_channels[3],    # x_3 channels
            self.branch_channels[1],      # Match stage 4 output
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        
        # PAG modules
        self.pag_1 = PagFM(self.branch_channels[0], self.branch_channels[0] // 2)
        self.pag_2 = PagFM(self.branch_channels[1], self.branch_channels[1] // 2)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function."""
        _, x_1, x_2, x_3, _, _ = x

        # stage 3: output shape (N, branch_channels[0], H/8, W/8)
        x_p = self.p_branch_layers[0](x_1)
        comp_i = self.compression_1(x_2)
        x_p = self.pag_1(x_p, comp_i)
        if self.training:
            temp_p = x_p.clone()

        # stage 4: output shape (N, branch_channels[1], H/8, W/8)
        x_p = self.p_branch_layers[1](self.relu(x_p))
        comp_i = self.compression_2(x_3)
        x_p = self.pag_2(x_p, comp_i)

        # stage 5: output shape (N, branch_channels[2], H/8, W/8)
        x_p = self.p_branch_layers[2](self.relu(x_p))
        
        return tuple([temp_p, x_p]) if self.training else x_p


class EdgeModuleScaled(CustomBaseModule):
    """
    Model layers for the D branch of PIDNet.
    """
    arch_settings = {
        'small': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [64, 128, 256],  # [stage3_out, stage4_out, stage5_out]
            'depths': [1, 1, 1],
            'stage_blocks': ['basic', 'bottleneck', 'bottleneck'],
        },
        'small_scaled': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [64, 128, 256],
            'depths': [1, 1, 1],
            'stage_blocks': ['basic', 'bottleneck', 'bottleneck'],
        },
        'small_constant': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [128, 128, 128],
            'depths': [1, 1, 1],
            'stage_blocks': ['basic', 'basic', 'bottleneck'],
        },
        'base': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [128, 128, 256],
            'depths': [1, 1, 1],
            'stage_blocks': ['basic', 'basic', 'bottleneck'],
        },
        'large': {
            'backbone_channels': [64, 128, 256, 512, 1024],
            'branch_channels': [128, 128, 256],
            'depths': [1, 1, 1],
            'stage_blocks': ['basic', 'basic', 'bottleneck'],
        },
        'xlarge': {
            'backbone_channels': [96, 192, 384, 768, 1536],
            'branch_channels': [192, 192, 384],
            'depths': [1, 1, 1],
            'stage_blocks': ['basic', 'basic', 'bottleneck'],
        },
    }

    def __init__(self,
                 arch: Union[str, dict] = 'small',
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='SyncBN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 eval_edges: bool = False,
                 **kwargs):
        super().__init__(init_cfg)
        
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Arch "{arch}" not found. Choose from {set(self.arch_settings.keys())} ' \
                f'or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            required = ['backbone_channels', 'branch_channels', 'depths', 'stage_blocks']
            assert all(k in arch for k in required), \
                f'Custom arch dict must have {required}.'
            assert len(arch['branch_channels']) == 3, \
                f'branch_channels must have 3 elements, got {len(arch["branch_channels"])}.'
        
        self.backbone_channels = arch['backbone_channels']
        self.branch_channels = arch['branch_channels']
        self.depths = arch['depths']
        self.stage_blocks = arch['stage_blocks']
        
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.eval_edges = eval_edges
        self.relu = nn.ReLU()

        # Helper to get block class
        def get_block(name):
            return BasicBlock if name == 'basic' else Bottleneck

        # D Branch layers
        self.d_branch_layers = nn.ModuleList()
        for i in range(3):
            block = get_block(self.stage_blocks[i])
            
            # Determine input channels
            if i == 0:
                in_ch = self.backbone_channels[1]  # x_1 channels
            else:
                in_ch = self.branch_channels[i - 1]
            
            # Determine layer channels (accounting for expansion)
            out_ch = self.branch_channels[i]
            layer_ch = out_ch // block.expansion
            
            # Use _make_single_layer only for BasicBlock with depth=1
            # This matches EdgeModuleFused behavior
            if self.depths[i] == 1 and self.stage_blocks[i] == 'basic':
                self.d_branch_layers.append(
                    self._make_single_layer(block, in_ch, layer_ch))
            else:
                self.d_branch_layers.append(
                    self._make_layer(block, in_ch, layer_ch, self.depths[i]))

        # Diff layers (backbone -> branch channels)
        self.diff_1 = ConvModule(
            self.backbone_channels[2],    # x_2 channels
            self.branch_channels[0],      # Match stage 3 output
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_2 = ConvModule(
            self.backbone_channels[3],    # x_3 channels
            self.branch_channels[1],      # Match stage 4 output
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function."""
        _, x_1, x_2, x_3, _, _ = x
        h_out, w_out = x_1.shape[-2:]

        # stage 3: output shape (N, branch_channels[0], H/8, W/8)
        x_d = self.d_branch_layers[0](x_1)
        diff_i = self.diff_1(x_2)
        x_d = x_d + F.interpolate(
            diff_i, size=[h_out, w_out], mode='bilinear', align_corners=self.align_corners)

        # stage 4: output shape (N, branch_channels[1], H/8, W/8)
        x_d = self.d_branch_layers[1](self.relu(x_d))
        diff_i = self.diff_2(x_3)
        x_d = x_d + F.interpolate(
            diff_i, size=[h_out, w_out], mode='bilinear', align_corners=self.align_corners)
        
        if self.training or self.eval_edges:
            temp_d = x_d.clone()

        # stage 5: output shape (N, branch_channels[2], H/8, W/8)
        x_d = self.d_branch_layers[2](self.relu(x_d))
        
        if self.training:
            return tuple([temp_d, x_d])
        return temp_d if self.eval_edges else x_d

class PModuleFused(CustomBaseModule):
    '''
    Model layers for the P branch of PIDNet. 

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
    '''
    # Optionally add argument `train` to constructor and pass `self.training` to it from the appropriate head module.
    # Another option is to register these modules if you need to.
    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.align_corners = align_corners

        self.relu = nn.ReLU()

        # P Branch
        self.p_branch_layers = nn.ModuleList()
        for i in range(3):
            self.p_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2,
                    channels=channels * 2,
                    num_blocks=num_stem_blocks if i < 2 else 1))
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.pag_1 = PagFM(channels * 2, channels)
        self.pag_2 = PagFM(channels * 2, channels)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.
        
        NOTE: self.training is inherent to MMSeg configurations throughout BaseModule 
        and BaseDecodeHead objects. Its boolean is inherited based on whether the
        train loop or the test/val loops are executing.
        
        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        
        """
        _, x_1, x_2, x_3, _, _ = x # x_0, x_1, x_2, x_3, x_4, x_out = x

        # stage 3
        x_p = self.p_branch_layers[0](x_1)

        comp_i = self.compression_1(x_2)
        x_p = self.pag_1(x_p, comp_i)
        if self.training:
            temp_p = x_p.clone() # (N, 128, H/8, W/8)

        # stage 4
        x_p = self.p_branch_layers[1](self.relu(x_p))

        comp_i = self.compression_2(x_3)
        x_p = self.pag_2(x_p, comp_i)

        # stage 5
        x_p = self.p_branch_layers[2](self.relu(x_p)) # (N, 256, H/8, W/8)
        
        return tuple([temp_p, x_p]) if self.training else x_p

class EdgeModuleFused(CustomBaseModule):
    '''
    Model layers for the D branch of PIDNet.
    '''
    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 eval_edges: bool = False,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.eval_edges = eval_edges
        self.relu = nn.ReLU()

        # D Branch
        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1)
            ])
            channel_expand = 1
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2,
                                        channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2)
            ])
            channel_expand = 2

        self.diff_1 = ConvModule(
            channels * 4,
            channels * channel_expand,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.d_branch_layers.append(
            self._make_layer(Bottleneck, channels * 2, channels * 2, 1))

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
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
        _, x_1, x_2, x_3, _, _ = x

        w_out = x[1].shape[-1]
        h_out = x[1].shape[-2]


        # stage 3
        x_d = self.d_branch_layers[0](x_1)

        diff_i = self.diff_1(x_2)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)

        # stage 4
        x_d = self.d_branch_layers[1](self.relu(x_d))

        diff_i = self.diff_2(x_3)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training or self.eval_edges:
            temp_d = x_d.clone()

        # stage 5
        x_d = self.d_branch_layers[2](self.relu(x_d))
        if self.training:
            return tuple([temp_d, x_d]) # temp_d: (N, 128, H/8, W/8), x_d: (N, 256, H/8, W/8)
        else:
            if self.eval_edges:
                return temp_d
            else:
                return x_d
    
class PModuleConditioned_Pag1(CustomBaseModule):
    '''
    Model layers for the P branch of PIDNet. 

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
    '''
    # Optionally add argument `train` to constructor and pass `self.training` to it from the appropriate head module.
    # Another option is to register these modules if you need to.
    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.align_corners = align_corners

        self.relu = nn.ReLU()

        # P Branch
        self.p_branch_layers = nn.ModuleList()
        for i in range(1):
            self.p_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2,
                    channels=channels * 2,
                    num_blocks=num_stem_blocks if i < 2 else 1))
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.pag_1 = PagFM(channels * 2, channels)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.
        
        NOTE: self.training is inherent to MMSeg configurations throughout BaseModule 
        and BaseDecodeHead objects. Its boolean is inherited based on whether the
        train loop or the test/val loops are executing.
        
        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        
        """
        _, x_1, x_2, x_3, _, _ = x # x_0, x_1, x_2, x_3, x_4, x_out = x

        # stage 3
        x_p = self.p_branch_layers[0](x_1)

        comp_i = self.compression_1(x_2)
        x_p = self.pag_1(x_p, comp_i)
        #if self.training:
            #temp_p = x_p.clone() # (N, 128, H/8, W/8)
        
        return tuple([x_p])
    
class PModuleConditioned_Pag2(CustomBaseModule):
    '''
    Model layers for the P branch of PIDNet. 

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
    '''
    # Optionally add argument `train` to constructor and pass `self.training` to it from the appropriate head module.
    # Another option is to register these modules if you need to.
    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 init_cfg: OptConfigType = None,
                 eval_edges: bool = False,
                 **kwargs):
        super().__init__(init_cfg)
        self.align_corners = align_corners

        self.relu = nn.ReLU()
        self.eval_edges = eval_edges
        # P Branch
        self.p_branch_layers = nn.ModuleList()
        for i in range(2):
            self.p_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2,
                    channels=channels * 2,
                    num_blocks=num_stem_blocks if i < 2 else 1))
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.pag_1 = PagFM(channels * 2, channels)
        self.pag_2 = PagFM(channels * 2, channels)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.
        
        NOTE: self.training is inherent to MMSeg configurations throughout BaseModule 
        and BaseDecodeHead objects. Its boolean is inherited based on whether the
        train loop or the test/val loops are executing.
        
        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        
        """
        _, x_1, x_2, x_3, _, _ = x # x_0, x_1, x_2, x_3, x_4, x_out = x

        # stage 3
        x_p = self.p_branch_layers[0](x_1)

        comp_i = self.compression_1(x_2)
        x_p = self.pag_1(x_p, comp_i)
        #if self.training:
        #    temp_p = x_p.clone() # (N, 128, H/8, W/8)

        # stage 4
        x_p = self.p_branch_layers[1](self.relu(x_p))

        comp_i = self.compression_2(x_3)
        x_p = self.pag_2(x_p, comp_i)
        
        if self.training or self.eval_edges:
            temp_p = x_p.clone()

        return temp_p
    
class PModuleConditioned_LastLayer(CustomBaseModule):
    '''
    Model layers for the P branch of PIDNet. 

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
    '''
    # Optionally add argument `train` to constructor and pass `self.training` to it from the appropriate head module.
    # Another option is to register these modules if you need to.
    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.align_corners = align_corners

        self.relu = nn.ReLU()

        # P Branch
        self.p_branch_layers = nn.ModuleList()
        for i in range(3):
            self.p_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2,
                    channels=channels * 2,
                    num_blocks=num_stem_blocks if i < 2 else 1))
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.pag_1 = PagFM(channels * 2, channels)
        self.pag_2 = PagFM(channels * 2, channels)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.
        
        NOTE: self.training is inherent to MMSeg configurations throughout BaseModule 
        and BaseDecodeHead objects. Its boolean is inherited based on whether the
        train loop or the test/val loops are executing.
        
        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        
        """
        _, x_1, x_2, x_3, _, _ = x # x_0, x_1, x_2, x_3, x_4, x_out = x

        # stage 3
        x_p = self.p_branch_layers[0](x_1)

        comp_i = self.compression_1(x_2)
        x_p = self.pag_1(x_p, comp_i)
        #if self.training:
        #    temp_p = x_p.clone() # (N, 128, H/8, W/8)

        # stage 4
        x_p = self.p_branch_layers[1](self.relu(x_p))

        comp_i = self.compression_2(x_3)
        x_p = self.pag_2(x_p, comp_i)

        # stage 5
        x_p = self.p_branch_layers[2](self.relu(x_p)) # (N, 256, H/8, W/8)
        
        return tuple([x_p])


class EdgeModuleFused(CustomBaseModule):
    '''
    Model layers for the D branch of PIDNet.
    '''
    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 eval_edges: bool = False,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.eval_edges = eval_edges
        self.relu = nn.ReLU()

        # D Branch
        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1)
            ])
            channel_expand = 1
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2,
                                        channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2)
            ])
            channel_expand = 2

        self.diff_1 = ConvModule(
            channels * 4,
            channels * channel_expand,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.d_branch_layers.append(
            self._make_layer(Bottleneck, channels * 2, channels * 2, 1))

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
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
        _, x_1, x_2, x_3, _, _ = x

        w_out = x[1].shape[-1]
        h_out = x[1].shape[-2]


        # stage 3
        x_d = self.d_branch_layers[0](x_1)

        diff_i = self.diff_1(x_2)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)

        # stage 4
        x_d = self.d_branch_layers[1](self.relu(x_d))

        diff_i = self.diff_2(x_3)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training or self.eval_edges:
            temp_d = x_d.clone()

        # stage 5
        x_d = self.d_branch_layers[2](self.relu(x_d))
        if self.training:
            return tuple([temp_d, x_d]) # temp_d: (N, 128, H/8, W/8), x_d: (N, 256, H/8, W/8)
        else:
            if self.eval_edges:
                return temp_d
            else:
                return x_d

class EdgeModuleConditioned(CustomBaseModule):
    '''
    Model layers for the D branch of PIDNet.
    '''
    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 eval_edges: bool = False,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.eval_edges = eval_edges
        self.relu = nn.ReLU()

        # D Branch
        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1)
            ])
            channel_expand = 1
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2,
                                        channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2)
            ])
            channel_expand = 2

        self.diff_1 = ConvModule(
            channels * 4,
            channels * channel_expand,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)


    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
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
        _, x_1, x_2, x_3, _, _ = x

        w_out = x[1].shape[-1]
        h_out = x[1].shape[-2]


        # stage 3
        x_d = self.d_branch_layers[0](x_1)

        diff_i = self.diff_1(x_2)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)

        # stage 4
        x_d = self.d_branch_layers[1](self.relu(x_d))

        diff_i = self.diff_2(x_3)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training or self.eval_edges:
            temp_d = x_d.clone()

        return temp_d # temp_d: (N, 128, H/8, W/8)

class CASENet(CustomBaseModule):
    '''
    Model layers for the CASENet SBD module.
    Slight changes to the CASENet architecture:
        1. Reduced resolution for consistency with PIDNet's D Branch
        2. Replaced nn.ConvTranspose2d() with F.interpolate to prevent checkerboarding
    '''
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CASENet, self).__init__(nclass, norm_layer=norm_layer, **kwargs)

        self.side1 = nn.Conv2d(64, 1, 1, stride=2, bias=True)
        self.side2 = nn.Conv2d(128, 1, 1, bias=True)
        self.side3 = nn.Conv2d(256, 1, 1, bias=True)
        self.side5 = nn.Conv2d(1024, nclass, 1, bias=True) # originally, 1024 was 2048; changed due to PIDNet architecture
        self.fuse = nn.Conv2d(nclass*4, nclass, 1, groups=nclass, bias=True)

    def forward(self, x):
        '''
        x should be a tuple of outputs:
        x_0, x_1, x_2, x_3, x_4, x_out = x
        x_0 has shape (N, 64, H/4, W/4)
        x_1 has shape (N, 128, H/8, W/8)
        x_2 has shape (N, 256, H/16, W/16)
        x_3 has shape (N, 512, H/32, W/32)
        x_4 has shape (N, 1024, H/64, W/64)
        x_out has shape (N, 256, H/64, W/64)
        '''
        c1, c2, c3, _, c5, _ = x
        height, width = c2.shape[2:]
        side1 = self.side1(c1) # (N, 1, H/8, W/8)
        side2 = self.side2(c2) # (N, 1, H/8, W/8)
        side3 = F.interpolate(self.side3(c3), # (N, 1, H/8, W/8)
                              size=[height, width],
                              mode='bilinear', align_corners=False)
        side5 = F.interpolate(self.side5(c5), # (N, K, H/8, W/8), where K is the number of classes in the labeled dataset
                              size=[height, width],
                              mode='bilinear', align_corners=False)
        slice5 = side5[:,0:1,:,:]
        fuse = torch.cat((slice5, side1, side2, side3), 1)
        for i in range(side5.size(1)-1):
            slice5 = side5[:,i+1:i+2,:,:]
            fuse = torch.cat((fuse, slice5, side1, side2, side3), 1)

        fuse = self.fuse(fuse)

        return tuple([side5, fuse]) if self.training else fuse
    
class DFF(CustomBaseModule):
    '''
    Model layers for the Dynamic Feature Fusion (DFF) SBD module.
    Slight changes to the DFF architecture:
        1. Reduced resolution for consistency with PIDNet's D Branch
        2. Replaced nn.ConvTranspose2d() with F.interpolate to prevent checkerboarding
    '''
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DFF, self).__init__(nclass, norm_layer=norm_layer, **kwargs)
        self.nclass = nclass
        self.ada_learner = LocationAdaptiveLearner(nclass, nclass*4, nclass*4, norm_layer=norm_layer)
        self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1, stride=2, bias=True),
                                   norm_layer(1))
        self.side2 = nn.Sequential(nn.Conv2d(128, 1, 1, bias=True),
                                   norm_layer(1))
        self.side3 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   norm_layer(1))
        self.side5 = nn.Sequential(nn.Conv2d(1024, nclass, 1, bias=True), # originally, 1024 was 2048; changed due to PIDNet architecture
                                   norm_layer(nclass))
        self.side5_w = nn.Sequential(nn.Conv2d(1024, nclass*4, 1, bias=True), # originally, 1024 was 2048; changed due to PIDNet architecture
                                   norm_layer(nclass*4))
        
    def forward(self, x):
        '''
        x should be a tuple of outputs:
        x_0, x_1, x_2, x_3, x_4, x_out = x
        x_0 has shape (N, 64, H/4, W/4)
        x_1 has shape (N, 128, H/8, W/8)
        x_2 has shape (N, 256, H/16, W/16)
        x_3 has shape (N, 512, H/32, W/32)
        x_4 has shape (N, 1024, H/64, W/64)
        x_out has shape (N, 256, H/64, W/64)
        '''
        c1, c2, c3, _, c5, _ = x
        height, width = c2.shape[2:]
        side1 = self.side1(c1) # (N, 1, H/8, W/8)
        side2 = self.side2(c2) # (N, 1, H/8, W/8)
        side3 = F.interpolate(self.side3(c3), # (N, 1, H/8, W/8)
                              size=[height, width],
                              mode='bilinear', align_corners=False)
        side5 = F.interpolate(self.side5(c5), # (N, K, H/8, W/8), where K is the number of classes in the labeled dataset
                              size=[height, width],
                              mode='bilinear', align_corners=False)
        side5_w = F.interpolate(self.side5_w(c5), # (N, K*4, H/8, W/8)
                                size=[height, width],
                                mode='bilinear', align_corners=False)
        
        ada_weights = self.ada_learner(side5_w) # (N, K, 4, H/8, W/8)

        slice5 = side5[:,0:1,:,:] # (N, 1, H/8, W/8)
        fuse = torch.cat((slice5, side1, side2, side3), 1)
        for i in range(side5.size(1)-1):
            slice5 = side5[:,i+1:i+2,:,:] # (N, 1, H/8, W/8)
            fuse = torch.cat((fuse, slice5, side1, side2, side3), 1) # (N, K*4, H/8, W/8)

        fuse = fuse.view(fuse.size(0), self.nclass, -1, fuse.size(2), fuse.size(3)) # (N, K, 4, H/8, W/8)
        fuse = torch.mul(fuse, ada_weights) # (N, K, 4, H/8, W/8)
        fuse = torch.sum(fuse, 2) # (N, K, H/8, W/8)

        return tuple([side5, fuse]) if self.training else fuse
    
class BEM(CustomBaseModule):
    '''
    Model layers for DCBNetv1's SBD module, Boundary Extraction Module (BEM).
    '''
    def __init__(self, planes=64, norm_layer=nn.BatchNorm2d, **kwargs):
        super(BEM, self).__init__(planes, norm_layer=norm_layer, **kwargs)
        self.norm_layer = norm_layer

        self.side1 = nn.Sequential(nn.Conv2d(in_channels=planes, out_channels=planes*2, kernel_size=3, stride=2, padding=1, bias=True), # (N, 128, H/4, W/4)
                                    self.norm_layer(num_features=planes*2))
        self.side2 = nn.Sequential(nn.Conv2d(in_channels=planes*2, out_channels=planes*2, kernel_size=3, padding=1, bias=True), # (N, C=128, H/4, W/4)
                                    self.norm_layer(num_features=planes*2))
        self.side3 = nn.Sequential(nn.Conv2d(in_channels=planes*4, out_channels=planes*2, kernel_size=3, padding=1, bias=True), # (N, C=128, H/4, W/4)
                                    self.norm_layer(num_features=planes*2))
        self.side5 = nn.Sequential(nn.Conv2d(in_channels=planes*16, out_channels=planes*2, kernel_size=3, padding=1, bias=True), # (N, C=128, H/4, W/4)
                                    self.norm_layer(num_features=planes*2))
        self.side5_w = nn.Sequential(nn.Conv2d(in_channels=planes*16, out_channels=planes*8, kernel_size=3, padding=1, bias=True), # (N, C=128*4, H/4, W/4)
                                    self.norm_layer(num_features=planes*8))

        self.layer1 = self._make_single_layer(BasicBlock, planes * 2, planes * 2) 
        self.layer2 = self._make_single_layer(BasicBlock, planes * 2, planes * 2)

        # No ReLU because we want side5 and fuse to have similar sequences and consequent responses
        self.sep_conv = nn.Sequential(
            nn.Conv2d(in_channels=planes*8, out_channels=planes*8, kernel_size=3, padding=1, groups=planes*8, bias=True),
            nn.Conv2d(in_channels=planes*8, out_channels=planes*2, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_features=planes*2),
            nn.ReLU(inplace=True)
        )

        self.adaptive_learner = LocationAdaptiveLearner(planes*2, planes*8, planes*8, norm_layer=self.norm_layer)

    def forward(self, x):
        '''
        x should be a tuple of outputs:
        x_0, x_1, x_2, x_3, x_4, x_out = x
        x_0 has shape (N, 64, H/4, W/4)
        x_1 has shape (N, 128, H/8, W/8)
        x_2 has shape (N, 256, H/16, W/16)
        x_3 has shape (N, 512, H/32, W/32)
        x_4 has shape (N, 1024, H/64, W/64)
        x_out has shape (N, 256, H/64, W/64)
        '''
        c1, c2, c3, _, c5, _ = x
        height, width = c2.shape[2:]
        '''Stage 1'''
        Aside1 = self.side1(c1) # (N, 128, H/8, W/8), may need to clone input

        '''Stage 2'''
        Aside2 = self.side2(c2) # (N, 128, H/8, W/8)
        Aside2 = self.layer1(Aside1 + Aside2) # (N, 128, H/8, W/8)
        
        '''Stage 3'''
        Aside3 = F.interpolate(self.side3(c3), # (N, 128, H/8, W/8)
                               size=[height, width],
                               mode='bilinear', align_corners=False)
        Aside3 = self.layer2(Aside3 + Aside2) # (N, 128, H/8, W/8)
        
        '''Stage 5'''
        Aside5 = Aside3 + F.interpolate(self.side5(c5), # (N, 128, H/8, W/8)
                                        size=[height, width],
                                        mode='bilinear', align_corners=False)

        Aside5_w = F.interpolate(self.side5_w(c5), # (N, 512, H/8, W/8)
                        size=[height, width],
                        mode='bilinear', align_corners=False)
        
        '''Fuse Sides 1-3 and 5'''
        adaptive_weights = F.softmax(self.adaptive_learner(Aside5_w), dim=2) # (N, 128, 4, H/8, W/8), softmax forces learned weights of each Aside to be mutually exclusive along the fusion dimension.
        concat = torch.cat((Aside1, Aside2, Aside3, Aside5), dim=1) # (N, 512, H/8, W/8)
        edge_5d = concat.view(concat.size(0), -1, 4, concat.size(2), concat.size(3)) # (N, 128, 4, H/8, W/8)
        fuse = torch.mul(edge_5d, adaptive_weights) # (N, 128, 4, H/8, W/8)
        fuse = fuse.view(fuse.size(0), -1, fuse.size(3), fuse.size(4)) # (N, 512, H/8, W/8)
        fuse = self.sep_conv(fuse) # (N, 128, H/8, W/8)
        
        return tuple([Aside5, fuse]) if self.training else fuse

class MIMIR(CustomBaseModule):
    '''
    Multi-scale Inverted Module for Image Refinement:
    BEM using Inverted Residual ConvNeXt blocks for side branches and layers.
    Renamed MIMIR.
    '''
    def __init__(self, planes=64, norm_cfg=dict(type='LN2d', eps=1e-6), **kwargs):
        #super().__init__()
        super(MIMIR, self).__init__(planes, norm_cfg=norm_cfg, **kwargs)
        self.norm_cfg = norm_cfg

        # Side1: Downsampling projection + ConvNeXt block
        self.side1_down = nn.Sequential(
            build_norm_layer(self.norm_cfg, planes),
            nn.Conv2d(planes, planes*2, kernel_size=3, stride=2, padding=1)
        )
        self.side1_block = ConvNeXtBlock(in_channels=planes*2, norm_cfg=self.norm_cfg)

        # Side2: Direct ConvNeXt block (same channels)
        self.side2_block = ConvNeXtBlock(in_channels=planes*2, norm_cfg=self.norm_cfg)

        # Side3: Channel projection + ConvNeXt block
        self.side3_proj = nn.Sequential(
            build_norm_layer(self.norm_cfg, planes*4),
            nn.Conv2d(planes*4, planes*2, kernel_size=1)
        )
        self.side3_block = ConvNeXtBlock(in_channels=planes*2, norm_cfg=self.norm_cfg)

        # Side5: Channel projection + ConvNeXt block
        self.side5_proj = nn.Sequential(
            build_norm_layer(self.norm_cfg, planes*16),
            nn.Conv2d(planes*16, planes*2, kernel_size=1)
        )
        self.side5_block = ConvNeXtBlock(in_channels=planes*2, norm_cfg=self.norm_cfg)

        # Side5_w: Channel projection + ConvNeXt block
        self.side5_w_proj = nn.Sequential(
            build_norm_layer(self.norm_cfg, planes*16),
            nn.Conv2d(planes*16, planes*8, kernel_size=1)
        )
        self.side5_w_block = ConvNeXtBlock(in_channels=planes*8, norm_cfg=self.norm_cfg)

        # Replace BasicBlock with ConvNeXtBlock
        self.layer1 = ConvNeXtBlock(planes*2, norm_cfg=self.norm_cfg)
        self.layer2 = ConvNeXtBlock(planes*2, norm_cfg=self.norm_cfg)

        # Adapted sep_conv with new norm
        self.sep_conv = nn.Sequential(
            nn.Conv2d(in_channels=planes*8, out_channels=planes*8, kernel_size=3, padding=1, groups=planes*8, bias=True),
            nn.Conv2d(in_channels=planes*8, out_channels=planes*2, kernel_size=1, bias=True),
            build_norm_layer(self.norm_cfg, planes*2)[1],  # Extract the module
            nn.ReLU(inplace=True)
        )

        self.adaptive_learner = LocationAdaptiveLearnerLN(planes*2, planes*8, planes*8, norm_cfg=self.norm_cfg)

    def forward(self, x):
        c1, c2, c3, _, c5, _ = x

        '''Stage 1'''
        Aside1 = self.side1_block(self.side1_down(c1))  # (N, 128, H/8, W/8)

        '''Stage 2'''
        Aside2 = self.side2_block(c2)  # (N, 128, H/8, W/8)
        Aside2 = self.layer1(Aside1 + Aside2)  # (N, 128, H/8, W/8)
        height, width = Aside2.shape[2:]

        '''Stage 3'''
        Aside3_proj = self.side3_proj(c3)  # Project channels
        Aside3 = self.side3_block(Aside3_proj)  # (N, 128, H/16, W/16) -> but interpolate later
        Aside3 = F.interpolate(Aside3, size=[height, width], mode='bilinear', align_corners=False)
        Aside3 = self.layer2(Aside3 + Aside2)  # (N, 128, H/8, W/8)
        
        '''Stage 5'''
        Aside5_proj = self.side5_proj(c5)
        Aside5 = self.side5_block(Aside5_proj)
        Aside5 = F.interpolate(Aside5, size=[height, width], mode='bilinear', align_corners=False)
        Aside5 = Aside3 + Aside5  # (N, 128, H/8, W/8)

        Aside5_w_proj = self.side5_w_proj(c5)
        Aside5_w = self.side5_w_block(Aside5_w_proj)
        Aside5_w = F.interpolate(Aside5_w, size=[height, width], mode='bilinear', align_corners=False)
        
        '''Fuse Sides 1-3 and 5'''
        adaptive_weights = F.softmax(self.adaptive_learner(Aside5_w), dim=2)  # (N, 128, 4, H/8, W/8)
        concat = torch.cat((Aside1, Aside2, Aside3, Aside5), dim=1)  # (N, 512, H/8, W/8)
        edge_5d = concat.view(concat.size(0), -1, 4, concat.size(2), concat.size(3))  # (N, 128, 4, H/8, W/8)
        fuse = torch.mul(edge_5d, adaptive_weights)  # (N, 128, 4, H/8, W/8)
        fuse = fuse.view(fuse.size(0), -1, fuse.size(3), fuse.size(4))  # (N, 512, H/8, W/8)
        fuse = self.sep_conv(fuse)  # (N, 128, H/8, W/8)

        outputs = [Aside5, fuse]
        
        return tuple(outputs)

class EdgeModuleFused_EarlierLayers(CustomBaseModule):
    '''
    Model layers for the D branch of PIDNet.
    Difference being that we convolve earlier layers.
    '''
    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 eval_edges: bool = False,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.eval_edges = eval_edges

        self.relu = nn.ReLU()

        # D Branch
        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1)
            ])
            channel_expand = 1
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2,
                                        channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2)
            ])
            channel_expand = 2
        
        self.diff_0 = ConvModule(
            channels,
            channels * channel_expand,
            kernel_size=3, # optionally change to 1, with no padding for faster computation. Not much of a speedup though.
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_1 = ConvModule(
            channels * 2,
            channels * channel_expand,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_2 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.d_branch_layers.append(
            self._make_layer(Bottleneck, channels * 2, channels * 2, 1))

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
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
        x_0, x_1, x_2, x_3, _, _ = x

        w_out = x[1].shape[-1]
        h_out = x[1].shape[-2]

        # stage 3
        diff_i = self.diff_0(x_0)
        x_d = F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        x_d = self.d_branch_layers[0](x_d)

        diff_i = self.diff_1(x_1)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)

        # stage 4
        x_d = self.d_branch_layers[1](self.relu(x_d))

        diff_i = self.diff_2(x_2)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training or self.eval_edges:
            temp_d = x_d.clone()

        # stage 5
        x_d = self.d_branch_layers[2](self.relu(x_d))
        if self.training:
            return tuple([temp_d, x_d]) # temp_d: (N, 128, H/8, W/8), x_d: (N, 256, H/8, W/8)
        else:
            if self.eval_edges:
                return temp_d
            else:
                return x_d

class EdgeModuleConditioned_EarlierLayers(CustomBaseModule):
    '''
    Model layers for the D branch of PIDNet.
    Difference being that we convolve earlier layers.
    '''
    def __init__(self,
                 channels: int = 64,
                 num_stem_blocks: int = 2,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 eval_edges: bool = False,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.eval_edges = eval_edges

        self.relu = nn.ReLU()

        # D Branch
        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1)
            ])
            channel_expand = 1
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2,
                                        channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2)
            ])
            channel_expand = 2
        
        self.diff_0 = ConvModule(
            channels,
            channels * channel_expand,
            kernel_size=3, # optionally change to 1, with no padding for faster computation. Not much of a speedup though.
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_1 = ConvModule(
            channels * 2,
            channels * channel_expand,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_2 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
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
        x_0, x_1, x_2, x_3, _, _ = x

        w_out = x[1].shape[-1]
        h_out = x[1].shape[-2]

        # stage 3
        diff_i = self.diff_0(x_0)
        x_d = F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        x_d = self.d_branch_layers[0](x_d)

        diff_i = self.diff_1(x_1)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)

        # stage 4
        x_d = self.d_branch_layers[1](self.relu(x_d))

        diff_i = self.diff_2(x_2)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        #if self.training or self.eval_edges:
        #    temp_d = x_d.clone()

        return x_d # x_: (N, 128, H/8, W/8)

class CASENet_EarlierLayers(CustomBaseModule):
    '''
    Model layers for the CASENet SBD module.
    Slight change to the CASENet architecture:
        CASENet doesn't normally normalize after side convolutions; however,
        they were added to prevent vanishing gradients during multi-task
        training.
    '''
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(CASENet_EarlierLayers, self).__init__(nclass, norm_layer=norm_layer, **kwargs)

        self.side1 = nn.Conv2d(64, 1, 1)
        self.side2 = nn.Sequential(nn.Conv2d(128, 1, 1, bias=True),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(1024, nclass, 1, bias=True), # originally, 1024 was 2048; changed due to PIDNet architecture
                                   nn.ConvTranspose2d(nclass, nclass, 32, stride=16, padding=8, bias=False))
        self.fuse = nn.Conv2d(nclass*4, nclass, 1, groups=nclass, bias=True)

    def forward(self, x):
        '''
        x should be a tuple of outputs:
        x_0, x_1, x_2, x_3, x_4, x_out = x
        x_0 has shape (N, 64, H/4, W/4)
        x_1 has shape (N, 128, H/8, W/8)
        x_2 has shape (N, 256, H/16, W/16)
        x_3 has shape (N, 512, H/32, W/32)
        x_4 has shape (N, 1024, H/64, W/64)
        x_out has shape (N, 256, H/64, W/64)
        '''
        c1, c2, c3, _, c5, _ = x

        side1 = self.side1(c1) # (N, 1, H/4, W/4)
        side2 = self.side2(c2) # (N, 1, H/4, W/4)
        side3 = self.side3(c3) # (N, 1, H/4, W/4)
        side5 = self.side5(c5) # (N, K, H/4, W/4), where K is the number of classes in the labeled

        slice5 = side5[:,0:1,:,:]
        fuse = torch.cat((slice5, side1, side2, side3), 1)
        for i in range(side5.size(1)-1):
            slice5 = side5[:,i+1:i+2,:,:]
            fuse = torch.cat((fuse, slice5, side1, side2, side3), 1)

        fuse = self.fuse(fuse)

        return tuple([side5, fuse]) if self.training else fuse
    
class DFF_EarlierLayers(CustomBaseModule):
    '''
    Model layers for the Dynamic Feature Fusion (DFF) SBD module.
    '''
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DFF_EarlierLayers, self).__init__(nclass, norm_layer=norm_layer, **kwargs)
        self.nclass = nclass
        self.ada_learner = LocationAdaptiveLearner(nclass, nclass*4, nclass*4, norm_layer=norm_layer)
        self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   norm_layer(1))
        self.side2 = nn.Sequential(nn.Conv2d(128, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(1024, nclass, 1, bias=True), # originally, 1024 was 2048; changed due to PIDNet architecture
                                   norm_layer(nclass),
                                   nn.ConvTranspose2d(nclass, nclass, 32, stride=16, padding=8, bias=False))

        self.side5_w = nn.Sequential(nn.Conv2d(1024, nclass*4, 1, bias=True), # originally, 1024 was 2048; changed due to PIDNet architecture
                                   norm_layer(nclass*4),
                                   nn.ConvTranspose2d(nclass*4, nclass*4, 32, stride=16, padding=8, bias=False))

    def forward(self, x):
        '''
        x should be a tuple of outputs:
        x_0, x_1, x_2, x_3, x_4, x_out = x
        x_0 has shape (N, 64, H/4, W/4)
        x_1 has shape (N, 128, H/8, W/8)
        x_2 has shape (N, 256, H/16, W/16)
        x_3 has shape (N, 512, H/32, W/32)
        x_4 has shape (N, 1024, H/64, W/64)
        x_out has shape (N, 256, H/64, W/64)
        '''
        c1, c2, c3, _, c5, _ = x
        side1 = self.side1(c1) # (N, 1, H/4, W/4)
        side2 = self.side2(c2) # (N, 1, H/4, W/4)
        side3 = self.side3(c3) # (N, 1, H/4, W/4)
        side5 = self.side5(c5) # (N, K, H/4, W/4), where K is the number of classes in the labeled dataset
        side5_w = self.side5_w(c5) # (N, K*4, H/4, W/4)
        
        ada_weights = self.ada_learner(side5_w) # (N, K, 4, H/4, W/4)

        slice5 = side5[:,0:1,:,:] # (N, 1, H/4, W/4)
        fuse = torch.cat((slice5, side1, side2, side3), 1)
        for i in range(side5.size(1)-1):
            slice5 = side5[:,i+1:i+2,:,:] # (N, 1, H/4, W/4)
            fuse = torch.cat((fuse, slice5, side1, side2, side3), 1) # (N, K*4, H/4, W/4)

        fuse = fuse.view(fuse.size(0), self.nclass, -1, fuse.size(2), fuse.size(3)) # (N, K, 4, H/4, W/4)
        fuse = torch.mul(fuse, ada_weights) # (N, K, 4, H/4, W/4)
        fuse = torch.sum(fuse, 2) # (N, K, H/4, W/4)

        return tuple([side5, fuse]) if self.training else fuse


class BEM_EarlierLayers(CustomBaseModule):
    '''
    Model layers for DCBNetv1's SBD module, Boundary Extraction Module (BEM).
    '''
    def __init__(self, planes=64, norm_layer=nn.BatchNorm2d, **kwargs):
        super(BEM_EarlierLayers, self).__init__(planes, norm_layer=norm_layer, **kwargs)
        self.norm_layer = norm_layer

        self.side1 = nn.Sequential(nn.Conv2d(in_channels=planes, out_channels=planes*2, kernel_size=3, padding=1, bias=True), # (N, 128, H/4, W/4)
                                    self.norm_layer(num_features=planes*2))
        self.side2 = nn.Sequential(nn.Conv2d(in_channels=planes*2, out_channels=planes*2, kernel_size=3, padding=1, bias=True), # (N, C=128, H/4, W/4)
                                    self.norm_layer(num_features=planes*2))
        self.side3 = nn.Sequential(nn.Conv2d(in_channels=planes*4, out_channels=planes*2, kernel_size=3, padding=1, bias=True), # (N, C=128, H/4, W/4)
                                    self.norm_layer(num_features=planes*2))
        self.side5 = nn.Sequential(nn.Conv2d(in_channels=planes*16, out_channels=planes*2, kernel_size=3, padding=1, bias=True), # (N, C=128, H/4, W/4)
                                    self.norm_layer(num_features=planes*2))
        self.side5_w = nn.Sequential(nn.Conv2d(in_channels=planes*16, out_channels=planes*8, kernel_size=3, padding=1, bias=True), # (N, C=128*4, H/4, W/4)
                                    self.norm_layer(num_features=planes*8))

        self.layer1 = self._make_single_layer(BasicBlock, planes * 2, planes * 2) 
        self.layer2 = self._make_single_layer(BasicBlock, planes * 2, planes * 2)

        # No ReLU because we want side5 and fuse to have similar sequences and consequent responses
        self.sep_conv = nn.Sequential(
            nn.Conv2d(in_channels=planes*8, out_channels=planes*8, kernel_size=3, padding=1, groups=planes*8, bias=True),
            nn.Conv2d(in_channels=planes*8, out_channels=planes*2, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_features=planes*2),
            nn.ReLU(inplace=True)
        )

        self.adaptive_learner = LocationAdaptiveLearner(planes*2, planes*8, planes*8, norm_layer=self.norm_layer)

    def forward(self, x):
        '''
        x should be a tuple of outputs:
        x_0, x_1, x_2, x_3, x_4, x_out = x
        x_0 has shape (N, 64, H/4, W/4)
        x_1 has shape (N, 128, H/8, W/8)
        x_2 has shape (N, 256, H/16, W/16)
        x_3 has shape (N, 512, H/32, W/32)
        x_4 has shape (N, 1024, H/64, W/64)
        x_out has shape (N, 256, H/64, W/64)
        '''
        c1, c2, c3, _, c5, _ = x
        '''Stage 1'''
        Aside1 = self.side1(c1) # (N, 128, H/4, W/4), may need to clone input
        height, width = Aside1.shape[2:]

        '''Stage 2'''
        Aside2 = F.interpolate(self.side2(c2), # (N, 128, H/4, W/4)
                               size=[height, width],
                               mode='bilinear', align_corners=False)
        Aside2 = self.layer1(Aside1 + Aside2) # (N, 128, H/4, W/4)
        
        '''Stage 3'''
        Aside3 = F.interpolate(self.side3(c3), # (N, 128, H/4, W/4)
                               size=[height, width],
                               mode='bilinear', align_corners=False)
        Aside3 = self.layer2(Aside3 + Aside2) # (N, 128, H/4, W/4)
        
        '''Stage 5'''
        Aside5 = Aside3 + F.interpolate(self.side5(c5), # (N, 128, H/4, W/4)
                                        size=[height, width],
                                        mode='bilinear', align_corners=False)

        Aside5_w = F.interpolate(self.side5_w(c5), # (N, 512, H/4, W/4)
                        size=[height, width],
                        mode='bilinear', align_corners=False)
        
        '''Fuse Sides 1-3 and 5'''
        adaptive_weights = F.softmax(self.adaptive_learner(Aside5_w), dim=2) # (N, 128, 4, H/4, W/4), softmax forces learned weights of each Aside to be mutually exclusive along the fusion dimension.
        concat = torch.cat((Aside1, Aside2, Aside3, Aside5), dim=1) # (N, 512, H/4, W/4)
        edge_5d = concat.view(concat.size(0), -1, 4, concat.size(2), concat.size(3)) # (N, 128, 4, H/4, W/4)
        fuse = torch.mul(edge_5d, adaptive_weights) # (N, 128, 4, H/4, W/4)
        fuse = fuse.view(fuse.size(0), -1, fuse.size(3), fuse.size(4)) # (N, 512, H/4, W/4)
        fuse = self.sep_conv(fuse) # (N, 128, H/4, W/4)
        
        return tuple([Aside5, fuse]) if self.training else fuse
    
class MIMIR_EarlierLayers(CustomBaseModule):
    '''
    Multi-scale Inverted Module for Image Refinement:
    BEM using Inverted Residual ConvNeXt blocks for side branches and layers.
    Renamed MIMIR.
    '''
    def __init__(self, planes=64, norm_cfg=dict(type='mmpretrain.LN2d', eps=1e-6), **kwargs):
        super().__init__()
        self.norm_cfg = norm_cfg

        # Side1: Channel projection + ConvNeXt block
        self.side1_proj = nn.Sequential(
            build_norm_layer(self.norm_cfg, planes)[1],
            nn.Conv2d(planes, planes*2, kernel_size=1)
        )
        self.side1_block = ConvNeXtBlock(in_channels=planes*2, norm_cfg=self.norm_cfg)

        # Side2: Direct ConvNeXt block (same channels)
        self.side2_block = ConvNeXtBlock(in_channels=planes*2, norm_cfg=self.norm_cfg)

        # Side3: Channel projection + ConvNeXt block
        self.side3_proj = nn.Sequential(
            build_norm_layer(self.norm_cfg, planes*4)[1],
            nn.Conv2d(planes*4, planes*2, kernel_size=1)
        )
        self.side3_block = ConvNeXtBlock(in_channels=planes*2, norm_cfg=self.norm_cfg)

        # Side5: Channel projection + ConvNeXt block
        self.side5_proj = nn.Sequential(
            build_norm_layer(self.norm_cfg, planes*16)[1],
            nn.Conv2d(planes*16, planes*2, kernel_size=1)
        )
        self.side5_block = ConvNeXtBlock(in_channels=planes*2, norm_cfg=self.norm_cfg)

        # Side5_w: Channel projection + ConvNeXt block
        self.side5_w_proj = nn.Sequential(
            build_norm_layer(self.norm_cfg, planes*16)[1],
            nn.Conv2d(planes*16, planes*8, kernel_size=1)
        )
        self.side5_w_block = ConvNeXtBlock(in_channels=planes*8, norm_cfg=self.norm_cfg)

        # Replace BasicBlock with ConvNeXtBlock
        self.layer1 = ConvNeXtBlock(planes*2, norm_cfg=self.norm_cfg)
        self.layer2 = ConvNeXtBlock(planes*2, norm_cfg=self.norm_cfg)

        # Adapted sep_conv with new norm
        self.sep_conv = nn.Sequential(
            nn.Conv2d(in_channels=planes*8, out_channels=planes*8, kernel_size=3, padding=1, groups=planes*8, bias=True),
            nn.Conv2d(in_channels=planes*8, out_channels=planes*2, kernel_size=1, bias=True),
            build_norm_layer(self.norm_cfg, planes*2)[1],  # Extract the module
            nn.ReLU(inplace=True)
        )

        self.adaptive_learner = LocationAdaptiveLearnerLN(planes*2, planes*8, planes*8, norm_cfg=self.norm_cfg)

    def forward(self, x):
        '''
        x should be a tuple of outputs:
        x_0, x_1, x_2, x_3, x_4, x_out = x
        x_0 has shape (N, 64, H/4, W/4)
        x_1 has shape (N, 128, H/8, W/8)
        x_2 has shape (N, 256, H/16, W/16)
        x_3 has shape (N, 512, H/32, W/32)
        x_4 has shape (N, 1024, H/64, W/64)
        x_out has shape (N, 256, H/64, W/64)
        '''
        c1, c2, c3, _, c5, _ = x

        '''Stage 1'''
        Aside1_proj = self.side1_proj(c1)
        Aside1 = self.side1_block(Aside1_proj)  # (N, 128, H/4, W/4)
        height, width = Aside1.shape[2:]

        '''Stage 2'''
        Aside2 = self.side2_block(c2)  # (N, 128, H/4, W/4)
        Aside2 = F.interpolate(Aside2, size=[height, width], mode='bilinear', align_corners=False)
        Aside2 = self.layer1(Aside1 + Aside2)  # (N, 128, H/4, W/4)

        '''Stage 3'''
        Aside3_proj = self.side3_proj(c3)  # Project channels
        Aside3 = self.side3_block(Aside3_proj)  # (N, 128, H/16, W/16) -> but interpolate later
        Aside3 = F.interpolate(Aside3, size=[height, width], mode='bilinear', align_corners=False)
        Aside3 = self.layer2(Aside3 + Aside2)  # (N, 128, H/4, W/4)
        
        '''Stage 5'''
        Aside5_proj = self.side5_proj(c5)
        Aside5 = self.side5_block(Aside5_proj)
        Aside5 = F.interpolate(Aside5, size=[height, width], mode='bilinear', align_corners=False)
        Aside5 = Aside3 + Aside5  # (N, 128, H/4, W/4)

        Aside5_w_proj = self.side5_w_proj(c5)
        Aside5_w = self.side5_w_block(Aside5_w_proj)
        Aside5_w = F.interpolate(Aside5_w, size=[height, width], mode='bilinear', align_corners=False)
        
        '''Fuse Sides 1-3 and 5'''
        adaptive_weights = F.softmax(self.adaptive_learner(Aside5_w), dim=2)  # (N, 128, 4, H/4, W/4)
        concat = torch.cat((Aside1, Aside2, Aside3, Aside5), dim=1)  # (N, 512, H/4, W/4)
        edge_5d = concat.view(concat.size(0), -1, 4, concat.size(2), concat.size(3))  # (N, 128, 4, H/4, W/4)
        fuse = torch.mul(edge_5d, adaptive_weights)  # (N, 128, 4, H/4, W/4)
        fuse = fuse.view(fuse.size(0), -1, fuse.size(3), fuse.size(4))  # (N, 512, H/4, W/4)
        fuse = self.sep_conv(fuse)  # (N, 128, H/4, W/4)
        
        outputs = [Aside5, fuse]
        
        return tuple(outputs)

class LocationAdaptiveLearnerLN(nn.Module):
    """Adaptive weight learner with 2D Layer Normalization."""
    def __init__(self, nclass, in_channels, out_channels, norm_cfg=dict(type='LN2d', eps=1e-6)):
        super().__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True),
                                   build_norm_layer(norm_cfg, out_channels)[1],
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   build_norm_layer(norm_cfg, out_channels)[1],
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   build_norm_layer(norm_cfg, out_channels)[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), self.nclass, -1, x.size(2), x.size(3))
        return x

class LocationAdaptiveLearner(nn.Module):
    """docstring for LocationAdaptiveLearner"""
    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(LocationAdaptiveLearner, self).__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels))

    def forward(self, x):
        # x:side5_w (N, 19*4, H, W)
        x = self.conv1(x) # (N, 19*4, H, W)
        x = self.conv2(x) # (N, 19*4, H, W)
        x = self.conv3(x) # (N, 19*4, H, W)
        x = x.view(x.size(0), self.nclass, -1, x.size(2), x.size(3)) # (N, 19, 4, H, W)
        return x