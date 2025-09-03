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


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SBD predictions')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='the dir to save logs and models')
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
    cfg.work_dir = args.work_dir
    cfg.model.decode_head.eval_edges = True  # Ensure edge mode for inference
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Initialize model
    model = init_model(cfg, cfg.ckpt, device='cuda:0')
    model.eval()

    # Build dataset using registry
    dataset=dict(
        type=cfg.dataset_type,
        data_root=cfg.data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=cfg.test_pipeline)
    dataset = DATASETS.build(dataset) # 'CityscapesDataset'

    # Build sampler using registry
    sampler_cfg = dict(type='DefaultSampler', shuffle=False, dataset=dataset)
    sampler = DATA_SAMPLERS.build(sampler_cfg)

    # Build dataloader directly with PyTorch's DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=6,  # Adjust as needed
        num_workers=6,
        persistent_workers=True,
        sampler=sampler,
        collate_fn=pseudo_collate  # Added collate_fn to handle SegDataSample objects
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating Edge Predictions'):
            # Preprocess batch
            data = model.data_preprocessor(batch, False)
            
            # Extract height and width of original image
            ori_h, ori_w = data['data_samples'][0].ori_shape

            # Extract features (backbone + neck)
            feats = model.extract_feat(data['inputs'])
            
            # Get edge logits from decode head forward (assumes eval_edges=True)
            logits = model.decode_head.forward(feats)  # Shape: (N, C, H, W)
            
            # Apply sigmoid for multi-label probabilities
            probs = torch.sigmoid(logits)

            # Correspond probabilities shape with ground truth shape
            _, _, h, w = probs.shape
            if ori_h != h or ori_w != w:
                probs = F.interpolate(probs, size=(ori_h, ori_w),
                                      mode='bilinear', align_corners=True).cpu().numpy()
            
            # Get batch metadata for filenames
            data_samples = data['data_samples']
            batch_names = [sample.metainfo.get('ori_filename', sample.metainfo['img_path'].split('/')[-1]) for sample in data_samples]
            
            # Assuming single sparse prediction type
            sparse_pred_name = 'd_module'
            
            for pred, name in zip(probs, batch_names):
                for idx_cls, pred_cls in enumerate(pred):
                    category_dir = f"class_{str(idx_cls+1).zfill(3)}"
                    out_dir = os.path.join(cfg.work_dir, sparse_pred_name, category_dir)
                    os.makedirs(out_dir, exist_ok=True)
                    
                    # Adapt basename parsing (for Cityscapes, split on image suffix)
                    basename = name.split('_leftImg8bit')[0] if '_leftImg8bit' in name else name.split('.')[0]
                    
                    # Scale and save as uint16 grayscale
                    scaled_pred = (pred_cls * 65535).astype(np.uint16)
                    img = Image.fromarray(scaled_pred, mode='I;16')
                    img.save(os.path.join(out_dir, f"{basename}_SBD.png"))

if __name__ == "__main__":
    main()