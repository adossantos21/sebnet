# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Optional, Sequence, List

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
import torch
import matplotlib.pyplot as plt


@HOOKS.register_module()
class FeatureMapVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 img_name: str = "path/to/image.png",
                 rstrip: Optional[str] = None,
                 out_dir: str = None):
        assert osp.isfile(img_name), f"img_name variable path is '{img_name}'; however, this file does not exist on this disk."
        self.img_name = img_name
        self.rstrip = rstrip
        self.out_dir = out_dir

    def custom_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample], logits: dict) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        for idx in range(runner.train_dataloader.batch_size):
            img_path = data_batch['data_samples'][idx].img_path
            if img_path == self.img_name:
                self.save_feature_maps(runner, img_path, logits, idx)

    def save_feature_maps(self, runner: Runner, img_path: str, logits: List[torch.Tensor], idx: int):
        # Determine save directory
        feature_maps = "feature_maps"
        if self.out_dir is None:
            save_dir = osp.join(runner._log_dir, feature_maps)
        else:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            date = osp.basename(runner._log_dir.rstrip(osp.sep))
            self.out_dir = osp.join(self.out_dir, basename, date, feature_maps)
            save_dir = self.out_dir
        os.makedirs(save_dir, exist_ok=True)

        # Create save path with iteration number
        img_name, ext = osp.splitext(osp.basename(img_path))
        if self.rstrip is not None:
            img_name = img_name.rstrip(self.rstrip)
        for k, v in logits.items(): # assume multiple types of logits
            save_path = osp.join(
                save_dir, 
                f'Iter_{runner.iter}_{k}_{img_name}_featuremap{ext}'
            )
            # Extract raw, unnormalized logits and average across channels
            viz_fmap = torch.mean(v[idx], dim=0).cpu().detach().numpy() # assume v has shape (B, C, H, W)
            plt.imsave(save_path, viz_fmap, cmap='jet')
            runner.logger.info(
                f'{k} Feature maps saved to {save_path} at iteration {runner.iter}'
            )
