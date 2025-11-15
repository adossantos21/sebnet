import torch
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from mmengine.config import Config
from mmseg.apis import init_model
from mmseg.registry import DATASETS, DATA_SAMPLERS
from torch.utils.data import DataLoader
from mmseg.utils import register_all_modules

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.dataset import pseudo_collate  # Added import for pseudo_collate

from mmseg.registry import RUNNERS

# Cityscapes 19-class color map dictionary for trainIds
# Maps trainId indices to RGB color tuples
cityscapes_colormap_trainid = {
    0: (128, 64, 128),   # road
    1: (244, 35, 232),   # sidewalk
    2: (70, 70, 70),     # building
    3: (102, 102, 156),  # wall
    4: (190, 153, 153),  # fence
    5: (153, 153, 153),  # pole
    6: (250, 170, 30),   # traffic light
    7: (220, 220, 0),    # traffic sign
    8: (107, 142, 35),   # vegetation
    9: (152, 251, 152),  # terrain
    10: (70, 130, 180),  # sky
    11: (220, 20, 60),   # person
    12: (255, 0, 0),     # rider
    13: (0, 0, 142),     # car
    14: (0, 0, 70),      # truck
    15: (0, 60, 100),    # bus
    16: (0, 80, 100),    # train
    17: (0, 0, 230),     # motorcycle
    18: (119, 11, 32)    # bicycle
}

# Mapping from trainId to labelId
train_to_label = {
    0: 7,    # road
    1: 8,    # sidewalk
    2: 11,   # building
    3: 12,   # wall
    4: 13,   # fence
    5: 17,   # pole
    6: 19,   # traffic light
    7: 20,   # traffic sign
    8: 21,   # vegetation
    9: 22,   # terrain
    10: 23,  # sky
    11: 24,  # person
    12: 25,  # rider
    13: 26,  # car
    14: 27,  # truck
    15: 28,  # bus
    16: 31,  # train
    17: 32,  # motorcycle
    18: 33   # bicycle
}

# Cityscapes color map for labelIds
cityscapes_colormap_labelid = {train_to_label[k]: v for k, v in cityscapes_colormap_trainid.items()}

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Dense predictions')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='the dir to save logs and models')
    parser.add_argument('split', default='val', help='whether to use the val or test split')
    parser.add_argument('--cityscapes', action='store_true', default=False, help='whether cityscapes is the dataset')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def colorize_trainid(pred: torch.Tensor) -> torch.Tensor:
    B, H, W = pred.shape
    output = torch.full((B, 3, H, W), 255, dtype=torch.uint8)
    for k, v in cityscapes_colormap_trainid.items():
        mask = (pred == k)
        output[:, 0][mask] = v[0]
        output[:, 1][mask] = v[1]
        output[:, 2][mask] = v[2]
    return output

def colorize_labelid(pred: torch.Tensor) -> torch.Tensor:
    B, H, W = pred.shape
    output = torch.full((B, 3, H, W), 255, dtype=torch.uint8)
    for k, v in cityscapes_colormap_labelid.items():
        mask = (pred == k)
        output[:, 0][mask] = v[0]
        output[:, 1][mask] = v[1]
        output[:, 2][mask] = v[2]
    return output

def main():
    # Register MMSeg modules
    register_all_modules()

    # Load config and set eval_edges if not already
    args = parse_args()
    assert args.checkpoint is not None, f"Checkpoint argument should not be empty."
    assert args.work_dir is not None, f"Working directory argument should not be empty."

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.ckpt = args.checkpoint
    cfg.work_dir = os.path.join(args.work_dir, args.split)
    os.makedirs(cfg.work_dir, exist_ok=True)
    trainid_raw_dir = os.path.join(cfg.work_dir, "trainid_dense_preds")
    trainid_color_dir = os.path.join(cfg.work_dir, "trainid_colorized_preds")
    os.makedirs(trainid_raw_dir, exist_ok=True)
    os.makedirs(trainid_color_dir, exist_ok=True)
    if args.cityscapes:
        labelid_raw_dir = os.path.join(cfg.work_dir, "labelid_dense_preds")
        labelid_color_dir = os.path.join(cfg.work_dir, "labelid_colorized_preds")
        os.makedirs(labelid_raw_dir, exist_ok=True)
        os.makedirs(labelid_color_dir, exist_ok=True)
    cfg.model.decode_head.eval_edges = False  # Ensure edge mode is off for inference

    # Initialize model
    model = init_model(cfg, cfg.ckpt, device='cuda:0')
    model.eval()

    # Build test dataset and dataloader (assuming same test config as baseline)
    dataloader = Runner.build_dataloader(cfg.val_dataloader)

    # Create mapping tensor for trainId to labelId
    if args.cityscapes:
        mapping_tensor = torch.zeros(256, dtype=torch.long)  # Larger to handle potential unexpected values
        for t, l in train_to_label.items():
            mapping_tensor[t] = l
        mapping_tensor[19:] = 255  # Default to 255 for any unexpected classes

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating Dense Predictions'):
            # Preprocess the batch            
            data = model.data_preprocessor(batch, False)
            batch_img_metas = [sample.metainfo for sample in data['data_samples']]
            
            # Extract height and width of original image
            ori_h, ori_w = data['data_samples'][0].ori_shape

            # Extract dense logits
            dense_logits = model.inference(data['inputs'], batch_img_metas)

            # Argmax across channel dimension (trainIds: 0-18)
            dense_preds_trainid = torch.argmax(dense_logits, dim=1)
            dense_preds_trainid_rgb = colorize_trainid(dense_preds_trainid)

            # Remap to labelIds
            if args.cityscapes:
                dense_preds_labelid = mapping_tensor.to(dense_preds_trainid.device)[dense_preds_trainid]
                dense_preds_labelid_rgb = colorize_labelid(dense_preds_labelid)
            
            # Get batch metadata for filenames
            data_samples = data['data_samples']
            batch_names = [sample.metainfo.get('ori_filename', sample.metainfo['img_path'].split('/')[-1]) for sample in data_samples]

            # Save trainId raw preds
            for pred, name in zip(dense_preds_trainid, batch_names):
                basename = name.split('_leftImg8bit')[0] if '_leftImg8bit' in name else name.split('.')[0]
                raw_img_array = pred.cpu().numpy().astype(np.uint8)
                img = Image.fromarray(raw_img_array)
                img.save(os.path.join(trainid_raw_dir, f"{basename}.png"))
            
            # Save trainId colorized preds
            for pred, name in zip(dense_preds_trainid_rgb, batch_names):
                basename = name.split('_leftImg8bit')[0] if '_leftImg8bit' in name else name.split('.')[0]
                color_img_array = pred.permute(1, 2, 0).cpu().numpy()
                img = Image.fromarray(color_img_array)
                img.save(os.path.join(trainid_color_dir, f"{basename}.png"))
            
            if args.cityscapes:
                # Save labelId raw preds
                for pred, name in zip(dense_preds_labelid, batch_names):
                    basename = name.split('_leftImg8bit')[0] if '_leftImg8bit' in name else name.split('.')[0]
                    raw_img_array = pred.cpu().numpy().astype(np.uint8)
                    img = Image.fromarray(raw_img_array)
                    img.save(os.path.join(labelid_raw_dir, f"{basename}.png"))

                # Save labelId colorized preds
                for pred, name in zip(dense_preds_labelid_rgb, batch_names):
                    basename = name.split('_leftImg8bit')[0] if '_leftImg8bit' in name else name.split('.')[0]
                    color_img_array = pred.permute(1, 2, 0).cpu().numpy()
                    img = Image.fromarray(color_img_array)
                    img.save(os.path.join(labelid_color_dir, f"{basename}.png"))

        print(f"Unique values from last trainId colorized pred: {np.unique(color_img_array)}")
        print(f"Unique values from last trainId raw pred: {np.unique(raw_img_array)}")

if __name__ == "__main__":
    main()